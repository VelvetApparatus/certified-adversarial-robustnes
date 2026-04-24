import argparse

args = argparse.ArgumentParser()

model_group = args.add_argument_group("model")
model_group.add_argument("--name", type=str)
model_group.add_argument("--weights_path", type=str)

ds_group = args.add_argument_group("dataset")
ds_group.add_argument("--train", type=bool)
ds_group.add_argument("--root_dir", type=str)
ds_group.add_argument("--download", type=bool)

adversaries_group = args.add_argument_group("adversaries")

# PGD
pgd_group = adversaries_group.add_argument_group("PGD")
pgd_group.add_argument("--epsilon", type=float)
pgd_group.add_argument("--alpha", type=float)
pgd_group.add_argument("--steps", type=int)
pgd_group.add_argument("--loss_fn", type=str)
pgd_group.add_argument("--norm", type=str)

# FGSM
fgsm_group = adversaries_group.add_argument_group("FGSM")
fgsm_group.add_argument("--epsilon", type=float)
fgsm_group.add_argument("--loss_fn", type=str)

# StAdv
stdadv_group = adversaries_group.add_argument_group("StAdv")
stdadv_group.add_argument("alpha", type=float)
stdadv_group.add_argument("--steps", type=int)
stdadv_group.add_argument("--tau", type=float)
stdadv_group.add_argument("targeted", type=bool)


def get_args():
    return args.parse_args()
