## Build

### Configure & build

```bash
git clone <repo-url> Lucas-Kanade
cd Lucas-Kanade
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The executable is placed at `build/myKLT`.

### Run

The program reads frames from `data/associate.txt`. With the dataset already in
`data/`, just run:

```bash
./build/myKLT
```

A window pops up showing the tracked keypoints; press any key to advance to the
next frame.

### Notes

- `CMAKE_BUILD_TYPE` defaults to `Release` in `CMakeLists.txt`. Switch to
  `Debug` for symbol info and Eigen bounds-checking when debugging.
- The build defines `EIGEN_DEFAULT_TO_ROW_MAJOR` so `Eigen::MatrixXd` aligns
  with `cv::Mat` row-major storage. Keep this in mind when interfacing with
  third-party libraries that assume column-major Eigen matrices.

## Forward compositional running performance
```bash
# running with pyramid of 3 level
frame 1: tracked 154/170 keypointscost time: 0.156237 seconds.
frame 2: tracked 154/170 keypointscost time: 0.161494 seconds.
frame 3: tracked 149/170 keypointscost time: 0.159733 seconds.
frame 4: tracked 151/170 keypointscost time: 0.148579 seconds.
frame 5: tracked 151/170 keypointscost time: 0.150312 seconds.
frame 6: tracked 155/170 keypointscost time: 0.148142 seconds.
frame 7: tracked 154/170 keypointscost time: 0.14382 seconds.
frame 8: tracked 153/170 keypointscost time: 0.154692 seconds.
```