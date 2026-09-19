# data/

Runtime data. Two kinds of thing live here.

## Video footage

Put traffic clips here. `config/config.yaml` points at `data/sample_traffic.mp4`
by default, so the quickest start is to save your clip under exactly that name.

Download the sample used to develop this project:

    curl.exe -o data\sample_traffic.mp4 https://media.roboflow.com/supervision/video-examples/vehicles.mp4

Requirements for any clip you use instead:

- **The camera must be static.** Drone and handheld footage breaks tracking
  completely: if the frame moves, every vehicle registers as moving, so queue
  length reads zero forever and speeds are nonsense. Before using a clip, watch
  a lamp post or kerb — if it drifts, discard the clip.
- Elevated and looking along the road, the way a junction CCTV camera sits.
- 1080p is plenty. 4K only makes detection slower for no accuracy gain here.

Video files are gitignored — they are large and not source code.

## The database

`traffic.db` is created here by `alembic upgrade head`. It is also gitignored:
it is generated data, and committing it would put your local traffic history
into the repository.

To reset the database completely, delete `traffic.db` (plus any `traffic.db-wal`
and `traffic.db-shm` files beside it) and run `alembic upgrade head` again.
