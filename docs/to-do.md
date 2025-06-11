# TODO

### 03/06/2025
- [x] Sort out the final phase dict with a complete list of phases for Cat21 and Cat101
- [x] Make sure all of Cat21 phase map to the final phase dict
- [x] Implement a phase map across for Cat101 that map to the final phase dict
- [] Capture not initialised as part of Cat101 in the _load_samples funciton (Decided not to do this for the initial experiment...)
- [] Check what is happening to the final sample of cat101. Does a single entry mean it runs to the end while a duplicate entry marks the end of the final phase? Once done fix this logic...
- [] Ensure split.py can split across multiple datasets and not just the first dataset in the index
- [] Make sure the number of classess being trained by the model is dynamically chosen based on the data provided
- [] Use a notebook to explore Cat21 and Cat101 to make sure the frame data is correct
- [] Visit your training code and make sure you're capturing F1-score and other metrics
- [] Chat about the best way to save these metric over multiple experiments and implement it
- [] Run a training round on all the data we have so far
- [] Download Cataract1K 🤯