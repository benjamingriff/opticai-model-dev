PHASE_MAP = {
    "Antibiotikum": "Antibiotikum",
    "Hydrodissektion": "Hydrodissektion",
    "Incision": "Incision",
    "Irrigation-Aspiration": "Irrigation-Aspiration",
    "Kapselpolishing": "Kapselpolishing",
    "Linsenimplantation": "Linsenimplantation",
    "Phako": "Phako",
    "Rhexis": "Rhexis",
    "Tonisieren": "Tonisieren",
    "Visco-Absaugung": "Visco-Absaugung",
    "Viscoelasticum": "Viscoelasticum",
    "not_initialized": "Idle",
}


def normalize_phase(label):
    return PHASE_MAP.get(label)
