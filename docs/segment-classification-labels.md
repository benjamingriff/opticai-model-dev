# Cataract Surgery Phases

This document provides a unified list of cataract surgery phases, along with their descriptions and common ways of phrasing these phases across datasets (`cat101` and `cat21`).

---

## Phase Table

| **Phase Title**              | **Description**                                                                 | **Common Phrasing**                                                                 |
|------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| **Incision**                 | The initial incision to access the anterior chamber of the eye.                 | `Incision`                                                                        |
| **Viscous agent injection**  | Injection of a viscous agent to protect the cornea and stabilize the anterior chamber. | `Viscoelasticum`, `Viscous agent injection`                                       |
| **Rhexis**                   | Capsulorhexis: creating a circular opening in the lens capsule.                 | `Rhexis`, `Capsulorhexis`                                                         |
| **Hydrodissection**          | Hydrodissection: separating the lens nucleus from the capsule using fluid.      | `Hydrodissektion`, `Hydrodissection`                                              |
| **Phacoemulsification**      | Breaking up and removing the lens nucleus using ultrasound.                     | `Phako`, `Phacoemulsificiation`                                                   |
| **Irrigation and aspiration**| Removing remaining lens material using irrigation and aspiration.               | `Irrigation-Aspiration`, `Irrigation and aspiration`                              |
| **Capsule polishing**        | Polishing the capsule to remove residual lens epithelial cells.                 | `Kapselpolishing`, `Capsule polishing`                                            |
| **Lens implant setting-up**  | Implanting the intraocular lens (IOL) into the capsule.                         | `Linsenimplantation`, `Lens implant setting-up`                                   |
| **Viscous agent removal**    | Removing the viscous agent from the anterior chamber.                          | `Visco-Absaugung`, `Viscous agent removal`                                        |
| **Tonifying and antibiotics**| Tonifying the eye and applying antibiotics to prevent infection.               | `Antibiotikum`, `Tonisieren`, `Tonifying and antibiotics`                         |
| **Not-Initialized**          | Placeholder for uninitialized or invalid phases.                               | `not_initialized`, `Not-Initialized`                                              |

---

## Notes

1. **Unified Terminology**:
   - The **Phase Title** column represents the unified terminology used across datasets.
   - This ensures consistency and clarity when working with multiple datasets.

2. **Descriptions**:
   - The **Description** column provides a brief explanation of each phase.

3. **Common Phrasing**:
   - The **Common Phrasing** column lists the different ways these phases are referred to in the datasets (`cat101` and `cat21`).

---

### Example Usage

When normalizing labels from datasets, use the **Phase Title** as the target label and map the **Common Phrasing** terms to it using a phase map.