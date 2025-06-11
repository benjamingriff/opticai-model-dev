PHASES = [
    "Incision",  # The initial incision to access the anterior chamber of the eye
    "Viscous agent injection",  # Injection of a viscous agent to protect the cornea and stabilize the anterior chamber
    "Rhexis",  # Capsulorhexis: creating a circular opening in the lens capsule
    "Hydrodissection",  # Hydrodissection: separating the lens nucleus from the capsule using fluid
    "Phacoemulsification",  # Breaking up and removing the lens nucleus using ultrasound
    "Irrigation and aspiration",  # Removing remaining lens material using irrigation and aspiration
    "Capsule polishing",  # Polishing the capsule to remove residual lens epithelial cells
    "Lens implant setting-up",  # Implanting the intraocular lens (IOL) into the capsule
    "Viscous agent removal",  # Removing the viscous agent from the anterior chamber
    "Tonifying and antibiotics",  # Tonifying the eye and applying antibiotics to prevent infection
    "Idle",  # Placeholder for uninitialized phases, invalid phases, or phases where no work is happening
]

phase2idx = {label: i for i, label in enumerate(PHASES)}
idx2phases = {i: label for label, i in phase2idx.items()}
