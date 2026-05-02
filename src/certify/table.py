import numpy as np
import torch
from tqdm import tqdm

from src.robustness.macer import Smooth


def _radius_key(radius: float) -> str:
    """
    0.25 -> cert_acc_025
    0.50 -> cert_acc_050
    1.00 -> cert_acc_100
    """
    return f"cert_acc_{int(round(radius * 100)):03d}"


def certify(
        model,
        device,
        dataset,
        num_classes,
        mode: str = "hard",
        start_img: int = 0,
        num_img: int = 500,
        skip: int = 1,
        sigma: float = 0.25,
        N0: int = 100,
        N: int = 100000,
        alpha: float = 0.001,
        batch: int = 1000,
        verbose: bool = False,
        grid=(0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25),
        beta: float = 1.0,
):
    """
    Randomized smoothing certification.

    Returns:
        If mode != "both":
            {
                "mode": "hard" / "soft",
                "sigma": float,
                "num_img": int,
                "cert_acc_000": float,
                "cert_acc_025": float,
                "cert_acc_050": float,
                ...
                "avg_radius": float,
                "median_radius": float,
                "radii": list[float],
            }

        If mode == "both":
            {
                "hard": {...},
                "soft": {...}
            }
    """

    print(f"=== certify(N={N}, sigma={sigma}, mode={mode}) ===")

    model.eval()

    smoothed_net = Smooth(
        model,
        num_classes,
        sigma,
        device,
        mode,
        beta,
    )

    num_grid = len(grid)

    radius_hard = np.full((num_img,), -1.0, dtype=float)
    radius_soft = np.full((num_img,), -1.0, dtype=float)

    cnt_grid_hard = np.zeros((num_grid + 1,), dtype=int)
    cnt_grid_soft = np.zeros((num_grid + 1,), dtype=int)

    sum_radius_hard = 0.0
    sum_radius_soft = 0.0

    for i in tqdm(range(num_img), desc="certification"):
        img, target = dataset[start_img + i * skip]

        img = img.to(device)

        if torch.is_tensor(target):
            target = target.item()

        if mode == "both":
            pred_hard, r_hard, pred_soft, r_soft = smoothed_net.certify(
                img,
                N0,
                N,
                alpha,
                batch,
            )

            correct_hard = int(pred_hard == target)
            correct_soft = int(pred_soft == target)

            if verbose:
                print(
                    f"[{i}] "
                    f"hard_pred={pred_hard}, soft_pred={pred_soft}, "
                    f"target={target}, "
                    f"r_hard={r_hard}, r_soft={r_soft}"
                )

            if correct_hard:
                radius_hard[i] = r_hard
                cnt_grid_hard[0] += 1
                sum_radius_hard += r_hard

                for j, r in enumerate(grid):
                    if r_hard >= r:
                        cnt_grid_hard[j + 1] += 1

            if correct_soft:
                radius_soft[i] = r_soft
                cnt_grid_soft[0] += 1
                sum_radius_soft += r_soft

                for j, r in enumerate(grid):
                    if r_soft >= r:
                        cnt_grid_soft[j + 1] += 1

        else:
            prediction, radius = smoothed_net.certify(
                img,
                N0,
                N,
                alpha,
                batch,
            )

            correct = int(prediction == target)

            if verbose:
                print(
                    f"[{i}] "
                    f"pred={prediction}, target={target}, "
                    f"correct={correct}, radius={radius}"
                )

            if correct:
                radius_hard[i] = radius
                cnt_grid_hard[0] += 1
                sum_radius_hard += radius

                for j, r in enumerate(grid):
                    if radius >= r:
                        cnt_grid_hard[j + 1] += 1

    def build_metrics(
            mode_name: str,
            radii: np.ndarray,
            cnt_grid: np.ndarray,
            radius_sum: float,
    ) -> dict:
        valid_radii = radii[radii >= 0.0]

        metrics = {
            "mode": mode_name,
            "sigma": sigma,
            "num_img": num_img,
            "N0": N0,
            "N": N,
            "alpha": alpha,
            "batch": batch,

            # certified accuracy at radius 0.0
            # это просто доля корректно классифицированных объектов
            # с ненулевым/валидным сертификатом
            "cert_acc_000": cnt_grid[0] / num_img,

            # average certified radius over all tested images,
            # incorrect samples contribute 0 through radius_sum / num_img
            "avg_radius": radius_sum / num_img,

            # median only over correctly certified samples
            "median_radius": float(np.median(valid_radii)) if len(valid_radii) > 0 else 0.0,

            # useful for debugging or extra plots
            "radii": radii.tolist(),
        }

        for j, r in enumerate(grid):
            metrics[_radius_key(r)] = cnt_grid[j + 1] / num_img

        return metrics

    if mode == "both":
        return {
            "hard": build_metrics(
                mode_name="hard",
                radii=radius_hard,
                cnt_grid=cnt_grid_hard,
                radius_sum=sum_radius_hard,
            ),
            "soft": build_metrics(
                mode_name="soft",
                radii=radius_soft,
                cnt_grid=cnt_grid_soft,
                radius_sum=sum_radius_soft,
            ),
        }

    return build_metrics(
        mode_name=mode,
        radii=radius_hard,
        cnt_grid=cnt_grid_hard,
        radius_sum=sum_radius_hard,
    )
