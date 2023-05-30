import colorama
from colorama import Fore, Style

colorama.init()


def print_params_in_color(seed, cfg, steps, strength, base_img, mask_img, model):
    print(
        Fore.MAGENTA
        + f"SEED: {seed} "
        + Fore.CYAN
        + f"CFG: {cfg} "
        + Fore.LIGHTGREEN_EX
        + f"STEPS: {steps} "
        + Fore.LIGHTYELLOW_EX
        + f"STRENGTH: {strength} "
        + Fore.LIGHTWHITE_EX
        + f"BASE: {base_img} "
        + f"MASK: {mask_img} "
        + Fore.LIGHTBLACK_EX
        + f"MODEL: {model}"
        + Style.RESET_ALL
    )
