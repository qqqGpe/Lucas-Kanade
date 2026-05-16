## Build

### Configure & build

```bash
git clone <repo-url> Lucas-Kanade
cd Lucas-Kanade
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
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

## Running performance

Setup: `patch_size = 21`, `n_max_iteration = 10`, `max_pyramid_level = 3`,
170 keypoints, Release build, both trackers invoked through
`std::unique_ptr<KltTracker>` (same call site).

| Tracker               | Avg cost | Min     | Max     | Tracked (avg) |
| --------------------- | -------: | ------: | ------: | ------------: |
| Forward compositional | 206.7 ms | 193.3 ms | 236.7 ms |       152/170 |
| Inverse compositional |  19.3 ms |  16.6 ms |  28.0 ms |       152/170 |

Inverse composition is **~10.7× faster** than forward composition at the
same tracking quality. The win comes from moving the Hessian (and its
LDLT factor) and the per-pixel arithmetic out of the inner Gauss–Newton
loop, leaving only one warped-patch resample and a sum of `SD^T · error`
per iteration.

### Forward compositional log

```bash
# pyramid of 3 levels, tracking 200 points
frame 2: tracked 154/170 keypoints, cost time: 0.204423 seconds.
frame 3: tracked 149/170 keypoints, cost time: 0.204258 seconds.
frame 4: tracked 151/170 keypoints, cost time: 0.199546 seconds.
frame 5: tracked 151/170 keypoints, cost time: 0.198716 seconds.
frame 6: tracked 155/170 keypoints, cost time: 0.193349 seconds.
frame 7: tracked 154/170 keypoints, cost time: 0.236748 seconds.
frame 8: tracked 153/170 keypoints, cost time: 0.209478 seconds.
```

### Inverse compositional log

```bash
# pyramid of 3 levels, tracking 200 points
frame 2: tracked 155/170 keypoints, cost time: 0.0182974 seconds.
frame 3: tracked 150/170 keypoints, cost time: 0.0168429 seconds.
frame 4: tracked 153/170 keypoints, cost time: 0.0174670 seconds.
frame 5: tracked 153/170 keypoints, cost time: 0.0179645 seconds.
frame 6: tracked 153/170 keypoints, cost time: 0.0196211 seconds.
frame 7: tracked 151/170 keypoints, cost time: 0.0279604 seconds.
frame 8: tracked 154/170 keypoints, cost time: 0.0166191 seconds.
```