PHASE_MAP = {
    "Antibiotikum": "Tonifying and antibiotics",
    "Hydrodissektion": "Hydrodissection",
    "Incision": "Incision",
    "Irrigation-Aspiration": "Irrigation and aspiration",
    "Kapselpolishing": "Capsule polishing",
    "Linsenimplantation": "Lens implant setting-up",
    "Phako": "Phacoemulsification",
    "Rhexis": "Rhexis",
    "Tonisieren": "Tonifying and antibiotics",
    "Visco-Absaugung": "Viscous agent removal",
    "Viscoelasticum": "Viscous agent injection",
    "not_initialized": "Idle",
}


def normalise_phase(label):
    return PHASE_MAP.get(label)
