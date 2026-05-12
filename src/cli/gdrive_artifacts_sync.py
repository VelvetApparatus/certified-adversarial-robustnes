#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from tqdm import tqdm


# Полный доступ к Google Drive от имени твоего аккаунта через OAuth.
SCOPES = ["https://www.googleapis.com/auth/drive"]

DEFAULT_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".onnx",
    ".pkl",
    ".joblib",
    ".parquet",
    ".zip",
    ".tar",
    ".gz",
}


def get_config_value(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Берём значение сначала из env.
    Если у тебя есть src.config.secret, пробуем взять оттуда.
    """
    value = os.environ.get(name)
    if value:
        return value

    try:
        from src.config import secret

        return getattr(secret, name, default)
    except Exception:
        return default


GDRIVE_FOLDER_ID = get_config_value("GDRIVE_FOLDER_ID")
GOAUTH_KEY_PATH = get_config_value("GOAUTH_KEY_PATH", "creds/google-oauth.json")
GDRIVE_MANIFEST_PATH = get_config_value(
    "GDRIVE_MANIFEST_PATH",
    get_config_value("GSERVICE_KEY_PATH", "artifacts/gdrive_manifest.json"),
)


def get_token_path(oauth_client_path: str) -> Path:
    oauth_path = Path(oauth_client_path)
    return oauth_path.parent / "google-drive-token.json"


def validate_oauth_client_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"OAuth client JSON not found: {path}\n"
            f"Создай OAuth Client ID → Desktop app в Google Cloud Console "
            f"и положи JSON сюда."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "installed" not in data:
        raise ValueError(
            f"Файл {path} не похож на OAuth Desktop client JSON.\n"
            f"В нём должен быть top-level ключ 'installed'.\n"
            f"Если там есть 'type': 'service_account', это неправильный файл."
        )


def get_drive_service():
    oauth_client_path = Path(GOAUTH_KEY_PATH)
    token_path = get_token_path(GOAUTH_KEY_PATH)

    validate_oauth_client_file(oauth_client_path)

    creds = None

    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception:
            invalid_path = token_path.with_suffix(token_path.suffix + ".invalid")
            token_path.rename(invalid_path)
            print(f"Invalid token file moved to: {invalid_path}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(oauth_client_path),
                SCOPES,
            )
            creds = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)

        with token_path.open("w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 4) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"files": []}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def get_drive_file(service, file_id: str) -> dict:
    return service.files().get(
        fileId=file_id,
        fields="id, name, mimeType",
        supportsAllDrives=True,
    ).execute()


def find_drive_folder(
    service,
    parent_folder_id: str,
    folder_name: str,
) -> Optional[dict]:
    safe_name = escape_drive_query_value(folder_name)

    query = (
        f"name = '{safe_name}' "
        f"and '{parent_folder_id}' in parents "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )

    response = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        pageSize=1,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    files = response.get("files", [])
    return files[0] if files else None


def get_or_create_drive_folder(
    service,
    parent_folder_id: str,
    folder_name: str,
) -> str:
    existing = find_drive_folder(
        service=service,
        parent_folder_id=parent_folder_id,
        folder_name=folder_name,
    )

    if existing:
        return existing["id"]

    metadata = {
        "name": folder_name,
        "parents": [parent_folder_id],
        "mimeType": "application/vnd.google-apps.folder",
    }

    folder = service.files().create(
        body=metadata,
        fields="id, name",
        supportsAllDrives=True,
    ).execute()

    return folder["id"]


def get_or_create_drive_path(
    service,
    root_folder_id: str,
    relative_dir: Path,
) -> str:
    current_folder_id = root_folder_id

    for part in relative_dir.parts:
        if part in ("", "."):
            continue

        current_folder_id = get_or_create_drive_folder(
            service=service,
            parent_folder_id=current_folder_id,
            folder_name=part,
        )

    return current_folder_id


def find_drive_file_by_name_and_size(
    service,
    folder_id: str,
    filename: str,
    size_bytes: int,
) -> Optional[dict]:
    safe_name = escape_drive_query_value(filename)

    query = (
        f"name = '{safe_name}' "
        f"and '{folder_id}' in parents "
        f"and trashed = false"
    )

    response = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, size, mimeType, webViewLink, createdTime, appProperties)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    candidates = response.get("files", [])

    for item in candidates:
        drive_size = int(item.get("size", -1))
        if drive_size == size_bytes:
            return item

    return None


def find_drive_file_by_sha256(
    service,
    folder_id: str,
    sha256: str,
) -> Optional[dict]:
    safe_sha = escape_drive_query_value(sha256)

    query = (
        f"appProperties has {{ key='sha256' and value='{safe_sha}' }} "
        f"and '{folder_id}' in parents "
        f"and trashed = false"
    )

    response = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, size, mimeType, webViewLink, createdTime, appProperties)",
        pageSize=1,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    files = response.get("files", [])
    return files[0] if files else None


def upload_file(
    service,
    local_path: Path,
    folder_id: str,
    sha256: str,
    relative_path: Optional[str] = None,
) -> dict:
    app_properties = {
        "sha256": sha256,
        "local_filename": local_path.name,
        "artifact_kind": "ml-checkpoint",
    }

    if relative_path:
        app_properties["relative_path"] = relative_path

    metadata = {
        "name": local_path.name,
        "parents": [folder_id],
        "appProperties": app_properties,
    }

    media = MediaFileUpload(
        str(local_path),
        resumable=True,
    )

    request = service.files().create(
        body=metadata,
        media_body=media,
        fields="id, name, size, mimeType, webViewLink, createdTime, appProperties",
        supportsAllDrives=True,
    )

    response = None

    with tqdm(
        total=100,
        desc=f"Uploading {local_path.name}",
        unit="%",
        leave=False,
    ) as progress_bar:
        last_progress = 0

        while response is None:
            status, response = request.next_chunk()

            if status:
                current_progress = int(status.progress() * 100)
                progress_bar.update(current_progress - last_progress)
                last_progress = current_progress

        if last_progress < 100:
            progress_bar.update(100 - last_progress)

    return response


def manifest_has_file(
    manifest: dict,
    sha256: str,
) -> Optional[dict]:
    for item in manifest.get("files", []):
        if item.get("sha256") == sha256:
            return item

    return None


def add_or_update_manifest_record(
    manifest: dict,
    local_path: Path,
    drive_file: dict,
    sha256: str,
    root_folder_id: str,
    drive_folder_id: str,
    relative_path: Optional[str] = None,
) -> None:
    record = {
        "local_path": str(local_path),
        "relative_path": relative_path,
        "filename": local_path.name,
        "size_bytes": local_path.stat().st_size,
        "sha256": sha256,
        "drive_root_folder_id": root_folder_id,
        "drive_folder_id": drive_folder_id,
        "drive_file_id": drive_file["id"],
        "drive_name": drive_file["name"],
        "drive_web_view_link": drive_file.get("webViewLink"),
        "drive_created_time": drive_file.get("createdTime"),
    }

    files = manifest.setdefault("files", [])

    for index, item in enumerate(files):
        if item.get("sha256") == sha256:
            files[index] = record
            return

    files.append(record)


def iter_files_from_directory(
    directory: Path,
    extensions: set[str],
    recursive: bool,
) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"

    for path in directory.glob(pattern):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        yield path


def sync_file(
    service,
    local_path: Path,
    root_folder_id: str,
    target_folder_id: str,
    manifest: dict,
    force_sha_check: bool = False,
    relative_path: Optional[str] = None,
) -> str:
    if not local_path.exists():
        return f"missing: {local_path}"

    if not local_path.is_file():
        return f"not a file: {local_path}"

    size_bytes = local_path.stat().st_size

    existing = find_drive_file_by_name_and_size(
        service=service,
        folder_id=target_folder_id,
        filename=local_path.name,
        size_bytes=size_bytes,
    )

    if existing and not force_sha_check:
        sha256 = existing.get("appProperties", {}).get("sha256")

        if not sha256:
            sha256 = sha256_file(local_path)

        add_or_update_manifest_record(
            manifest=manifest,
            local_path=local_path,
            drive_file=existing,
            sha256=sha256,
            root_folder_id=root_folder_id,
            drive_folder_id=target_folder_id,
            relative_path=relative_path,
        )

        return f"skip, exists on Drive: {relative_path or local_path.name}"

    sha256 = sha256_file(local_path)

    manifest_record = manifest_has_file(manifest, sha256)
    if manifest_record:
        return f"skip, exists in manifest: {relative_path or local_path.name}"

    existing_by_sha = find_drive_file_by_sha256(
        service=service,
        folder_id=target_folder_id,
        sha256=sha256,
    )

    if existing_by_sha:
        add_or_update_manifest_record(
            manifest=manifest,
            local_path=local_path,
            drive_file=existing_by_sha,
            sha256=sha256,
            root_folder_id=root_folder_id,
            drive_folder_id=target_folder_id,
            relative_path=relative_path,
        )

        return f"skip, exists on Drive by sha256: {relative_path or local_path.name}"

    uploaded = upload_file(
        service=service,
        local_path=local_path,
        folder_id=target_folder_id,
        sha256=sha256,
        relative_path=relative_path,
    )

    add_or_update_manifest_record(
        manifest=manifest,
        local_path=local_path,
        drive_file=uploaded,
        sha256=sha256,
        root_folder_id=root_folder_id,
        drive_folder_id=target_folder_id,
        relative_path=relative_path,
    )

    return f"uploaded: {relative_path or local_path.name}"


def command_sync_dir(args):
    root_folder_id = args.folder_id or GDRIVE_FOLDER_ID

    if not root_folder_id:
        raise ValueError("folder_id is required. Use --folder-id or GDRIVE_FOLDER_ID.")

    service = get_drive_service()

    root_info = get_drive_file(service, root_folder_id)
    if root_info.get("mimeType") != "application/vnd.google-apps.folder":
        raise ValueError(f"Drive ID is not a folder: {root_folder_id}")

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.extensions
    }

    base_dir = Path(args.directory).resolve()

    files = list(
        iter_files_from_directory(
            directory=base_dir,
            extensions=extensions,
            recursive=args.recursive,
        )
    )

    print(f"Found {len(files)} candidate files")

    folder_cache: dict[str, str] = {
        ".": root_folder_id,
    }

    for path in files:
        local_path = path.resolve()
        relative_path_obj = local_path.relative_to(base_dir)
        relative_dir = relative_path_obj.parent
        relative_path_str = relative_path_obj.as_posix()

        relative_dir_key = relative_dir.as_posix()

        if relative_dir_key not in folder_cache:
            folder_cache[relative_dir_key] = get_or_create_drive_path(
                service=service,
                root_folder_id=root_folder_id,
                relative_dir=relative_dir,
            )

        target_folder_id = folder_cache[relative_dir_key]

        result = sync_file(
            service=service,
            local_path=local_path,
            root_folder_id=root_folder_id,
            target_folder_id=target_folder_id,
            manifest=manifest,
            force_sha_check=args.force_sha_check,
            relative_path=relative_path_str,
        )

        print(result)

    save_manifest(manifest_path, manifest)
    print(f"Manifest saved to {manifest_path}")


def command_sync_files(args):
    root_folder_id = args.folder_id or GDRIVE_FOLDER_ID

    if not root_folder_id:
        raise ValueError("folder_id is required. Use --folder-id or GDRIVE_FOLDER_ID.")

    service = get_drive_service()

    root_info = get_drive_file(service, root_folder_id)
    if root_info.get("mimeType") != "application/vnd.google-apps.folder":
        raise ValueError(f"Drive ID is not a folder: {root_folder_id}")

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    for raw_path in args.files:
        path = Path(raw_path).resolve()

        result = sync_file(
            service=service,
            local_path=path,
            root_folder_id=root_folder_id,
            target_folder_id=root_folder_id,
            manifest=manifest,
            force_sha_check=args.force_sha_check,
            relative_path=path.name,
        )

        print(result)

    save_manifest(manifest_path, manifest)
    print(f"Manifest saved to {manifest_path}")


def command_list(args):
    manifest = load_manifest(Path(args.manifest))

    files = manifest.get("files", [])

    if not files:
        print("Manifest is empty")
        return

    for item in files:
        print(
            f"{item.get('relative_path') or item['filename']} | "
            f"{item['size_bytes']} bytes | "
            f"{item['sha256'][:12]}... | "
            f"{item['drive_file_id']}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sync ML artifacts to Google Drive using OAuth.",
    )

    parser.add_argument(
        "--manifest",
        default=GDRIVE_MANIFEST_PATH or "artifacts/gdrive_manifest.json",
        help="Path to local manifest JSON.",
    )

    subparsers = parser.add_subparsers(required=True)

    sync_dir = subparsers.add_parser(
        "sync-dir",
        help="Sync all matching files from a directory.",
    )
    sync_dir.add_argument("directory")
    sync_dir.add_argument("--folder-id", default=GDRIVE_FOLDER_ID)
    sync_dir.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(DEFAULT_EXTENSIONS),
    )
    sync_dir.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directory recursively.",
    )
    sync_dir.add_argument(
        "--force-sha-check",
        action="store_true",
        help="Always calculate sha256 and check Drive appProperties.",
    )
    sync_dir.set_defaults(func=command_sync_dir)

    sync_files = subparsers.add_parser(
        "sync-files",
        help="Sync explicitly provided files.",
    )
    sync_files.add_argument("files", nargs="+")
    sync_files.add_argument("--folder-id", default=GDRIVE_FOLDER_ID)
    sync_files.add_argument(
        "--force-sha-check",
        action="store_true",
        help="Always calculate sha256 and check Drive appProperties.",
    )
    sync_files.set_defaults(func=command_sync_files)

    list_cmd = subparsers.add_parser(
        "list",
        help="List files from local manifest.",
    )
    list_cmd.set_defaults(func=command_list)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()