

def relative_ufs(clean_utility: float, attacked_utility: float) -> float:
    """Return 5 * f, where f is the non-negative relative utility drop."""
    if clean_utility <= 0:
        return 0
    if not 0 <= attacked_utility <= 1 or not 0 < clean_utility <= 1:
        raise ValueError(
            "Utilities used for UFS must be in [0, 1]; "
            f"clean={clean_utility:.6g}, attacked={attacked_utility:.6g}"
        )

    relative_drop = (clean_utility - attacked_utility) / clean_utility
    f_value = max(0.0, relative_drop)
    return 5 * f_value
