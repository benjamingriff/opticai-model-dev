PHASES = [
    "Incision",  # The initial incision to access the eye
    "Viscoelasticum",  # Injection of a viscous agent to protect the cornea and stabilize the anterior chamber
    "Rhexis",  # Capsulorhexis: creating a circular opening in the lens capsule
    "Hydrodissektion",  # Hydrodissection: separating the lens nucleus from the capsule
    "Phako",  # Phacoemulsification: breaking up and removing the lens nucleus
    "Irrigation-Aspiration",  # Removing remaining lens material using irrigation and aspiration
    "Kapselpolishing",  # Polishing the capsule to remove residual lens epithelial cells
    "Linsenimplantation",  # Implanting the intraocular lens (IOL)
    "Visco-Absaugung",  # Removing the viscous agent from the anterior chamber
    "Tonisieren",  # Tonifying the eye and applying antibiotics
    "Antibiotikum",  # Final application of antibiotics
    "Idle",  # Idle phases
]

phase2idx = {label: i for i, label in enumerate(PHASES)}
idx2phases = {i: label for label, i in phase2idx.items()}
