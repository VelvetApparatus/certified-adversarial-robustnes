from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from tqdm import tqdm

try:
    from src.config import secret
except Exception:  # pragma: no cover
    secret = None

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
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".log",
    ".txt",
}

SCOPES = ["https://www.googleapis.com/auth/drive"]
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DEFAULT_MANIFEST_PATH = "artifacts/gdrive_manifest.json"
DEFAULT_OAUTH_PATH = "creds/google-oauth.json"
DEFAULT_SERVICE_ACCOUNT_PATH = "creds/google-key.json"


def get_config_value(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    if secret is not None and hasattr(secret, name):
        value = getattr(secret, name)
        if value:
            return value
    return default


def get_manifest_path(args_manifest: Optional[str] = None) -> Path:
    value = args_manifest or get_config_value("GDRIVE_MANIFEST_PATH", DEFAULT_MANIFEST_PATH)
    return Path(value)


def get_root_folder_id(args_folder_id: Optional[str] = None) -> str:
    folder_id = (
            args_folder_id
            or get_config_value("GDRIVE_FOLDER_ID")
            or get_config_value("GOOGLE_DRIVE_FOLDER_ID")
            or ""
    )
    if not folder_id:
        raise ValueError(
            "Google Drive folder_id is not set. Pass --folder-id, set "
            "GDRIVE_FOLDER_ID, or set GOOGLE_DRIVE_FOLDER_ID."
        )
    return folder_id


def read_json_if_exists(path: str | Path | None) -> Optional[dict]:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def candidate_service_account_paths(explicit_path: Optional[str] = None) -> list[str]:
    candidates = []
    for value in [
        explicit_path,
        get_config_value("GOOGLE_SERVICE_ACCOUNT_JSON"),
        get_config_value("GDRIVE_SERVICE_ACCOUNT_PATH"),
        get_config_value("GSERVICE_KEY_PATH"),
        DEFAULT_SERVICE_ACCOUNT_PATH,
    ]:
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def candidate_oauth_paths(explicit_path: Optional[str] = None) -> list[str]:
    candidates = []
    for value in [
        explicit_path,
        get_config_value("GOAUTH_KEY_PATH"),
        DEFAULT_OAUTH_PATH,
    ]:
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def get_token_path(oauth_client_path: str | Path) -> Path:
    oauth_path = Path(oauth_client_path).expanduser()
    return oauth_path.parent / "google-drive-token.json"


def get_drive_service(
        *,
        auth: str = "auto",
        oauth_client_path: Optional[str] = None,
        service_account_path: Optional[str] = None,
):
    """Create Google Drive service.

    auth:
      - auto: prefer a valid service account json if provided; otherwise OAuth.
      - oauth: require OAuth Desktop client JSON.
      - service: require service account JSON.
    """
    auth = auth.lower()
    if auth not in {"auto", "oauth", "service"}:
        raise ValueError("auth must be one of: auto, oauth, service")

    if auth in {"auto", "service"}:
        for path in candidate_service_account_paths(service_account_path):
            data = read_json_if_exists(path)
            if not data:
                continue
            if data.get("type") != "service_account":
                continue
            credentials = service_account.Credentials.from_service_account_file(
                str(Path(path).expanduser()),
                scopes=SCOPES,
            )
            print(f"Using Google Drive service account credentials: {path}")
            return build("drive", "v3", credentials=credentials)
        if auth == "service":
            checked = ", ".join(candidate_service_account_paths(service_account_path))
            raise FileNotFoundError(
                "No valid service account JSON found. Checked: " + checked
            )

    # OAuth fallback / explicit OAuth.
    for path in candidate_oauth_paths(oauth_client_path):
        data = read_json_if_exists(path)
        if not data:
            continue
        if "installed" not in data:
            continue

        oauth_path = Path(path).expanduser()
        token_path = get_token_path(oauth_path)
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
                flow = InstalledAppFlow.from_client_secrets_file(str(oauth_path), SCOPES)
                creds = flow.run_local_server(port=0)

            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")

        print(f"Using Google Drive OAuth credentials: {oauth_path}")
        return build("drive", "v3", credentials=creds)

    checked_oauth = ", ".join(candidate_oauth_paths(oauth_client_path))
    checked_sa = ", ".join(candidate_service_account_paths(service_account_path))
    raise FileNotFoundError(
        "No valid Google credentials found. "
        f"OAuth checked: {checked_oauth}. Service account checked: {checked_sa}."
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 4) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"files": []}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if "files" not in data or not isinstance(data["files"], list):
        data["files"] = []
    return data


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def get_drive_file(service, file_id: str) -> dict:
    return service.files().get(
        fileId=file_id,
        fields="id, name, mimeType, size, webViewLink, createdTime, modifiedTime, appProperties, parents",
        supportsAllDrives=True,
    ).execute()


def list_drive_children(service, parent_folder_id: str, page_size: int = 1000) -> Iterable[dict]:
    query = f"'{parent_folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        response = service.files().list(
            q=query,
            spaces="drive",
            fields=(
                "nextPageToken, files(id, name, mimeType, size, webViewLink, "
                "createdTime, modifiedTime, appProperties, parents)"
            ),
            pageSize=page_size,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        yield from response.get("files", [])
        page_token = response.get("nextPageToken")
        if not page_token:
            break


def find_drive_folder(service, parent_folder_id: str, folder_name: str) -> Optional[dict]:
    safe_name = escape_drive_query_value(folder_name)
    query = (
        f"name = '{safe_name}' "
        f"and '{parent_folder_id}' in parents "
        f"and mimeType = '{DRIVE_FOLDER_MIME_TYPE}' "
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


def get_or_create_drive_folder(service, parent_folder_id: str, folder_name: str) -> str:
    existing = find_drive_folder(service, parent_folder_id, folder_name)
    if existing:
        return existing["id"]

    metadata = {
        "name": folder_name,
        "parents": [parent_folder_id],
        "mimeType": DRIVE_FOLDER_MIME_TYPE,
    }
    folder = service.files().create(
        body=metadata,
        fields="id, name",
        supportsAllDrives=True,
    ).execute()
    return folder["id"]


def get_or_create_drive_path(service, root_folder_id: str, relative_dir: Path) -> str:
    current_folder_id = root_folder_id
    for part in relative_dir.parts:
        if part in ("", "."):
            continue
        current_folder_id = get_or_create_drive_folder(service, current_folder_id, part)
    return current_folder_id


def find_drive_file_by_name_and_size(
        service,
        folder_id: str,
        filename: str,
        size_bytes: int,
) -> Optional[dict]:
    safe_name = escape_drive_query_value(filename)
    query = f"name = '{safe_name}' and '{folder_id}' in parents and trashed = false"
    response = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, size, mimeType, webViewLink, createdTime, modifiedTime, appProperties)",
        pageSize=20,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    for item in response.get("files", []):
        try:
            drive_size = int(item.get("size", -1))
        except (TypeError, ValueError):
            drive_size = -1
        if drive_size == size_bytes:
            return item
    return None


def find_drive_file_by_sha256(service, folder_id: str, sha256: str) -> Optional[dict]:
    safe_sha = escape_drive_query_value(sha256)
    query = (
        f"appProperties has {{ key='sha256' and value='{safe_sha}' }} "
        f"and '{folder_id}' in parents and trashed = false"
    )
    response = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, size, mimeType, webViewLink, createdTime, modifiedTime, appProperties)",
        pageSize=5,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = response.get("files", [])
    return files[0] if files else None


def normalize_extensions(raw_extensions: Iterable[str]) -> set[str]:
    return {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in raw_extensions
    }


def should_include_file(path_or_name: str | Path, extensions: set[str], include_all: bool) -> bool:
    if include_all:
        return True
    return Path(str(path_or_name)).suffix.lower() in extensions


def iter_files_from_directory(directory: Path, extensions: set[str], recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in directory.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        yield path


def upload_file(
        service,
        local_path: Path,
        folder_id: str,
        sha256: str,
        relative_path: Optional[Path] = None,
) -> dict:
    app_properties = {
        "sha256": sha256,
        "local_filename": local_path.name,
        "artifact_kind": "ml-artifact",
    }
    if relative_path is not None:
        app_properties["relative_path"] = relative_path.as_posix()

    metadata = {
        "name": local_path.name,
        "parents": [folder_id],
        "appProperties": app_properties,
    }
    media = MediaFileUpload(str(local_path), resumable=True)
    request = service.files().create(
        body=metadata,
        media_body=media,
        fields="id, name, size, mimeType, webViewLink, createdTime, modifiedTime, appProperties",
        supportsAllDrives=True,
    )

    response = None
    with tqdm(total=100, desc=f"Uploading {local_path.name}", unit="%", leave=False) as pbar:
        last_progress = 0
        while response is None:
            status, response = request.next_chunk()
            if status:
                current_progress = int(status.progress() * 100)
                pbar.update(current_progress - last_progress)
                last_progress = current_progress
        if last_progress < 100:
            pbar.update(100 - last_progress)
    return response


def download_drive_file(service, file_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with output_path.open("wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        with tqdm(total=100, desc=f"Downloading {output_path.name}", unit="%", leave=False) as pbar:
            last_progress = 0
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    current_progress = int(status.progress() * 100)
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
            if last_progress < 100:
                pbar.update(100 - last_progress)


def manifest_has_file(manifest: dict, sha256: str, relative_path: Optional[Path] = None) -> Optional[dict]:
    relative_path_str = relative_path.as_posix() if relative_path is not None else None
    for item in manifest.get("files", []):
        if item.get("sha256") != sha256:
            continue
        if relative_path_str is None:
            return item
        if item.get("relative_path") == relative_path_str:
            return item
    return None


def add_or_update_manifest_record(
        manifest: dict,
        *,
        local_path: Optional[Path],
        drive_file: dict,
        root_folder_id: str,
        drive_folder_id: str,
        relative_path: Optional[Path | str] = None,
        sha256: Optional[str] = None,
) -> None:
    if isinstance(relative_path, Path):
        relative_path_str = relative_path.as_posix()
    else:
        relative_path_str = relative_path

    size = drive_file.get("size")
    try:
        size_bytes = int(size) if size is not None else (local_path.stat().st_size if local_path else None)
    except Exception:
        size_bytes = None

    if sha256 is None:
        sha256 = (drive_file.get("appProperties") or {}).get("sha256")

    record = {
        "local_path": str(local_path) if local_path is not None else None,
        "relative_path": relative_path_str,
        "filename": drive_file.get("name") or (local_path.name if local_path else None),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "drive_root_folder_id": root_folder_id,
        "drive_folder_id": drive_folder_id,
        "drive_file_id": drive_file["id"],
        "drive_name": drive_file.get("name"),
        "drive_web_view_link": drive_file.get("webViewLink"),
        "drive_created_time": drive_file.get("createdTime"),
        "drive_modified_time": drive_file.get("modifiedTime"),
    }

    files = manifest.setdefault("files", [])
    for index, item in enumerate(files):
        same_drive_id = item.get("drive_file_id") == record["drive_file_id"]
        same_relative_path = relative_path_str is not None and item.get("relative_path") == relative_path_str
        same_sha_without_path = relative_path_str is None and sha256 and item.get("sha256") == sha256
        if same_drive_id or same_relative_path or same_sha_without_path:
            files[index] = record
            return
    files.append(record)


def sync_file(
        service,
        *,
        local_path: Path,
        folder_id: str,
        manifest: dict,
        force_sha_check: bool = False,
        root_folder_id: Optional[str] = None,
        relative_path: Optional[Path] = None,
) -> str:
    if not local_path.exists():
        return f"missing: {local_path}"
    if not local_path.is_file():
        return f"not a file: {local_path}"

    root_folder_id = root_folder_id or folder_id
    size_bytes = local_path.stat().st_size
    display_path = relative_path.as_posix() if relative_path else local_path.name

    existing = find_drive_file_by_name_and_size(service, folder_id, local_path.name, size_bytes)
    if existing and not force_sha_check:
        sha256 = (existing.get("appProperties") or {}).get("sha256") or sha256_file(local_path)
        add_or_update_manifest_record(
            manifest,
            local_path=local_path,
            drive_file=existing,
            sha256=sha256,
            drive_folder_id=folder_id,
            root_folder_id=root_folder_id,
            relative_path=relative_path,
        )
        return f"skip, exists on Drive: {display_path}"

    sha256 = sha256_file(local_path)

    if manifest_has_file(manifest, sha256=sha256, relative_path=relative_path):
        return f"skip, exists in manifest: {display_path}"

    existing_by_sha = find_drive_file_by_sha256(service, folder_id, sha256)
    if existing_by_sha:
        add_or_update_manifest_record(
            manifest,
            local_path=local_path,
            drive_file=existing_by_sha,
            sha256=sha256,
            drive_folder_id=folder_id,
            root_folder_id=root_folder_id,
            relative_path=relative_path,
        )
        return f"skip, exists on Drive by sha256: {display_path}"

    uploaded = upload_file(service, local_path, folder_id, sha256, relative_path)
    add_or_update_manifest_record(
        manifest,
        local_path=local_path,
        drive_file=uploaded,
        sha256=sha256,
        drive_folder_id=folder_id,
        root_folder_id=root_folder_id,
        relative_path=relative_path,
    )
    return f"uploaded: {display_path}"


def resolve_download_path(record: dict, output_dir: Path, use_local_path: bool = False) -> Path:
    if use_local_path and record.get("local_path"):
        return Path(record["local_path"])
    if record.get("relative_path"):
        return output_dir / record["relative_path"]
    return output_dir / (record.get("filename") or record.get("drive_name"))


def verify_downloaded_file(output_path: Path, expected_sha: Optional[str]) -> None:
    if not expected_sha:
        return
    actual_sha = sha256_file(output_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"sha256 mismatch for {output_path}: expected {expected_sha}, got {actual_sha}"
        )


def download_manifest_record(
        service,
        record: dict,
        *,
        output_dir: Path,
        overwrite: bool,
        use_local_path: bool = False,
) -> str:
    drive_file_id = record.get("drive_file_id")
    if not drive_file_id:
        raise ValueError(f"Manifest record is missing drive_file_id: {record}")

    output_path = resolve_download_path(record, output_dir=output_dir, use_local_path=use_local_path)
    expected_sha = record.get("sha256")
    expected_size = record.get("size_bytes")

    if output_path.exists():
        if output_path.is_dir():
            raise ValueError(f"Output path is a directory, expected file: {output_path}")
        if expected_sha:
            actual_sha = sha256_file(output_path)
            if actual_sha == expected_sha:
                return f"skip, already downloaded: {output_path}"
            if not overwrite:
                raise ValueError(
                    f"Existing file has wrong sha256: {output_path}. "
                    f"Expected {expected_sha}, got {actual_sha}. Use --overwrite to replace it."
                )
        elif expected_size is not None and output_path.stat().st_size == int(expected_size):
            return f"skip, already exists with same size: {output_path}"
        elif not overwrite:
            raise ValueError(
                f"Existing file cannot be verified and --overwrite is not set: {output_path}"
            )

    download_drive_file(service, drive_file_id, output_path)
    verify_downloaded_file(output_path, expected_sha)
    return f"downloaded: {output_path}"


def select_manifest_records(
        manifest: dict,
        *,
        sha256_values: Optional[list[str]] = None,
        drive_file_ids: Optional[list[str]] = None,
        filenames: Optional[list[str]] = None,
        relative_paths: Optional[list[str]] = None,
) -> list[dict]:
    sha256_set = set(sha256_values or [])
    drive_file_id_set = set(drive_file_ids or [])
    filename_set = set(filenames or [])
    relative_path_set = set(relative_paths or [])
    if not any((sha256_set, drive_file_id_set, filename_set, relative_path_set)):
        raise ValueError(
            "At least one selector must be provided: --sha256, --drive-file-id, --filename, or --relative-path."
        )

    selected = []
    seen = set()
    for record in manifest.get("files", []):
        matches = (
                record.get("sha256") in sha256_set
                or record.get("drive_file_id") in drive_file_id_set
                or record.get("filename") in filename_set
                or record.get("drive_name") in filename_set
                or record.get("relative_path") in relative_path_set
        )
        if not matches:
            continue
        key = record.get("drive_file_id") or record.get("relative_path") or record.get("sha256")
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
    return selected


def build_manifest_from_drive_folder(
        service,
        *,
        root_folder_id: str,
        manifest: dict,
        extensions: set[str],
        recursive: bool,
        include_all: bool,
        current_folder_id: Optional[str] = None,
        current_relative_dir: Path = Path("."),
) -> int:
    current_folder_id = current_folder_id or root_folder_id
    count = 0

    for item in list_drive_children(service, current_folder_id):
        name = item["name"]
        mime = item.get("mimeType")
        rel_path = current_relative_dir / name

        if mime == DRIVE_FOLDER_MIME_TYPE:
            if recursive:
                count += build_manifest_from_drive_folder(
                    service,
                    root_folder_id=root_folder_id,
                    manifest=manifest,
                    extensions=extensions,
                    recursive=recursive,
                    include_all=include_all,
                    current_folder_id=item["id"],
                    current_relative_dir=rel_path,
                )
            continue

        if not should_include_file(name, extensions, include_all):
            continue

        add_or_update_manifest_record(
            manifest,
            local_path=None,
            drive_file=item,
            root_folder_id=root_folder_id,
            drive_folder_id=current_folder_id,
            relative_path=rel_path,
            sha256=(item.get("appProperties") or {}).get("sha256"),
        )
        count += 1

    return count


def make_service_from_args(args):
    return get_drive_service(
        auth=getattr(args, "auth", "auto"),
        oauth_client_path=getattr(args, "oauth-client", None) if False else getattr(args, "oauth_client", None),
        service_account_path=getattr(args, "service_account", None),
    )


def command_sync_dir(args):
    service = make_service_from_args(args)
    root_folder_id = get_root_folder_id(args.folder_id)
    root_info = get_drive_file(service, root_folder_id)
    if root_info.get("mimeType") != DRIVE_FOLDER_MIME_TYPE:
        raise ValueError(f"Drive ID is not a folder: {root_folder_id}")

    manifest_path = get_manifest_path(args.manifest)
    manifest = load_manifest(manifest_path)
    base_dir = Path(args.directory).resolve()
    extensions = normalize_extensions(args.extensions)
    files = list(iter_files_from_directory(base_dir, extensions, args.recursive))
    print(f"Found {len(files)} candidate files")

    folder_cache: dict[str, str] = {".": root_folder_id}
    for path in files:
        local_path = path.resolve()
        relative_path = local_path.relative_to(base_dir)
        relative_dir = relative_path.parent
        relative_dir_key = relative_dir.as_posix()
        if relative_dir_key not in folder_cache:
            folder_cache[relative_dir_key] = get_or_create_drive_path(
                service, root_folder_id, relative_dir
            )
        result = sync_file(
            service,
            local_path=local_path,
            folder_id=folder_cache[relative_dir_key],
            root_folder_id=root_folder_id,
            relative_path=relative_path,
            manifest=manifest,
            force_sha_check=args.force_sha_check,
        )
        print(result)

    save_manifest(manifest_path, manifest)
    print(f"Manifest saved to {manifest_path}")


def command_sync_files(args):
    service = make_service_from_args(args)
    root_folder_id = get_root_folder_id(args.folder_id)
    root_info = get_drive_file(service, root_folder_id)
    if root_info.get("mimeType") != DRIVE_FOLDER_MIME_TYPE:
        raise ValueError(f"Drive ID is not a folder: {root_folder_id}")

    manifest_path = get_manifest_path(args.manifest)
    manifest = load_manifest(manifest_path)
    base_dir = Path(args.base_dir).resolve() if args.base_dir else None
    folder_cache: dict[str, str] = {".": root_folder_id}

    for raw_path in args.files:
        local_path = Path(raw_path).resolve()
        if base_dir is not None:
            try:
                relative_path = local_path.relative_to(base_dir)
            except ValueError as exc:
                raise ValueError(f"File is not inside --base-dir: {local_path} (base={base_dir})") from exc
            relative_dir = relative_path.parent
            relative_dir_key = relative_dir.as_posix()
            if relative_dir_key not in folder_cache:
                folder_cache[relative_dir_key] = get_or_create_drive_path(
                    service, root_folder_id, relative_dir
                )
            target_folder_id = folder_cache[relative_dir_key]
        else:
            relative_path = Path(local_path.name)
            target_folder_id = root_folder_id

        result = sync_file(
            service,
            local_path=local_path,
            folder_id=target_folder_id,
            root_folder_id=root_folder_id,
            relative_path=relative_path,
            manifest=manifest,
            force_sha_check=args.force_sha_check,
        )
        print(result)

    save_manifest(manifest_path, manifest)
    print(f"Manifest saved to {manifest_path}")


def command_build_manifest(args):
    service = make_service_from_args(args)
    root_folder_id = get_root_folder_id(args.folder_id)
    root_info = get_drive_file(service, root_folder_id)
    if root_info.get("mimeType") != DRIVE_FOLDER_MIME_TYPE:
        raise ValueError(f"Drive ID is not a folder: {root_folder_id}")

    manifest_path = get_manifest_path(args.manifest)
    manifest = {"files": []} if args.replace else load_manifest(manifest_path)
    extensions = normalize_extensions(args.extensions)
    count = build_manifest_from_drive_folder(
        service,
        root_folder_id=root_folder_id,
        manifest=manifest,
        extensions=extensions,
        recursive=args.recursive,
        include_all=args.all_files,
    )
    save_manifest(manifest_path, manifest)
    print(f"Added/updated {count} records from Drive folder {root_folder_id}")
    print(f"Manifest saved to {manifest_path}")


def command_download_all(args):
    service = make_service_from_args(args)
    manifest = load_manifest(get_manifest_path(args.manifest))
    records = manifest.get("files", [])
    output_dir = Path(args.output_dir)
    if not records:
        print("Manifest is empty")
        return

    downloaded = 0
    skipped = 0
    for record in records:
        result = download_manifest_record(
            service,
            record,
            output_dir=output_dir,
            overwrite=args.overwrite,
            use_local_path=args.use_local_path,
        )
        print(result)
        if result.startswith("downloaded:"):
            downloaded += 1
        else:
            skipped += 1
    print(f"Download summary: total={len(records)}, downloaded={downloaded}, skipped={skipped}")


def command_download(args):
    service = make_service_from_args(args)
    manifest = load_manifest(get_manifest_path(args.manifest))
    output_dir = Path(args.output_dir)
    records = select_manifest_records(
        manifest,
        sha256_values=args.sha256,
        drive_file_ids=args.drive_file_id,
        filenames=args.filename,
        relative_paths=args.relative_path,
    )
    if not records:
        raise ValueError("No manifest records matched the provided selectors.")

    downloaded = 0
    skipped = 0
    for record in records:
        result = download_manifest_record(
            service,
            record,
            output_dir=output_dir,
            overwrite=args.overwrite,
            use_local_path=args.use_local_path,
        )
        print(result)
        if result.startswith("downloaded:"):
            downloaded += 1
        else:
            skipped += 1
    print(f"Download summary: total={len(records)}, downloaded={downloaded}, skipped={skipped}")


def command_list(args):
    manifest = load_manifest(get_manifest_path(args.manifest))
    files = manifest.get("files", [])
    if not files:
        print("Manifest is empty")
        return
    for item in files:
        display_path = item.get("relative_path") or item.get("filename") or item.get("drive_name")
        sha = item.get("sha256") or "no-sha"
        size = item.get("size_bytes")
        print(f"{display_path} | {size} bytes | {sha[:12]}... | {item.get('drive_file_id')}")


def add_common_auth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auth",
        choices=["auto", "oauth", "service"],
        default="auto",
        help="Google auth mode. auto prefers valid service account JSON if present, otherwise OAuth.",
    )
    parser.add_argument("--oauth-client", default=None, help="OAuth Desktop client JSON path.")
    parser.add_argument("--service-account", default=None, help="Service account JSON path.")


def build_parser():
    parser = argparse.ArgumentParser(description="Sync ML artifacts to/from Google Drive without Git LFS.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to local manifest JSON. Defaults to GDRIVE_MANIFEST_PATH or artifacts/gdrive_manifest.json.",
    )
    add_common_auth_args(parser)

    subparsers = parser.add_subparsers(required=True)

    sync_dir = subparsers.add_parser("sync-dir",
                                     help="Upload matching files from a directory; preserve relative subdirectories on Drive.")
    sync_dir.add_argument("directory")
    sync_dir.add_argument("--folder-id", default=None)
    sync_dir.add_argument("--extensions", nargs="+", default=sorted(DEFAULT_EXTENSIONS))
    sync_dir.add_argument("--recursive", action="store_true")
    sync_dir.add_argument("--force-sha-check", action="store_true")
    sync_dir.set_defaults(func=command_sync_dir)

    sync_files = subparsers.add_parser("sync-files", help="Upload explicitly provided files.")
    sync_files.add_argument("files", nargs="+")
    sync_files.add_argument("--folder-id", default=None)
    sync_files.add_argument("--base-dir", default=None, help="Preserve paths relative to this directory on Drive.")
    sync_files.add_argument("--force-sha-check", action="store_true")
    sync_files.set_defaults(func=command_sync_files)

    build_manifest = subparsers.add_parser("build-manifest",
                                           help="Build/update local manifest by scanning a Drive folder.")
    build_manifest.add_argument("--folder-id", default=None)
    build_manifest.add_argument("--extensions", nargs="+", default=sorted(DEFAULT_EXTENSIONS))
    build_manifest.add_argument("--recursive", action="store_true")
    build_manifest.add_argument("--all-files", action="store_true", help="Ignore extension filtering.")
    build_manifest.add_argument("--replace", action="store_true",
                                help="Replace existing manifest instead of updating it.")
    build_manifest.set_defaults(func=command_build_manifest)

    list_cmd = subparsers.add_parser("list", help="List files from local manifest.")
    list_cmd.set_defaults(func=command_list)

    download_all = subparsers.add_parser("download-all", help="Download every file referenced in local manifest.")
    download_all.add_argument("--output-dir", default=".")
    download_all.add_argument("--overwrite", action="store_true")
    download_all.add_argument("--use-local-path", action="store_true")
    download_all.set_defaults(func=command_download_all)

    download = subparsers.add_parser("download", help="Download selected files from local manifest.")
    download.add_argument("--output-dir", default=".")
    download.add_argument("--overwrite", action="store_true")
    download.add_argument("--use-local-path", action="store_true")
    download.add_argument("--sha256", nargs="*")
    download.add_argument("--drive-file-id", nargs="*", dest="drive_file_id")
    download.add_argument("--filename", nargs="*")
    download.add_argument("--relative-path", nargs="*", dest="relative_path")
    download.set_defaults(func=command_download)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
