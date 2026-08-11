# Jetson AGX Thor R39.2 Migration Gate

## Why the migration is required

The installed Jetson Linux R38.4 stack exposes the Thor GPU's MIG mode but does
not support the MIG instance workflow needed by this project. JetPack 7.2,
which contains Jetson Linux R39.2, enables MIG on the T5000 as a technology
preview. The research baseline therefore requires R39.2 or newer.

This board was upgraded in place from R38.4 to R39.2 using NVIDIA's documented
APT minor-release procedure: the Jetson repository was changed to `r39.2`, the
package index was refreshed, and a full distribution upgrade was installed.
The board then booted the R39.2 kernel successfully. An ISO or initrd flash
remains the recovery path when the package upgrade is not viable.

## Pre-upgrade gate

1. Attach and mount an external disk with enough free capacity.
2. Capture the repository, uncommitted files, results, package manifest, and
   platform state:

   ```bash
   ./scripts/capture_upgrade_state.sh /media/thor/BACKUP_DISK
   ```

3. Run the printed `sha256sum -c` command from the backup directory.
4. Confirm that the presentation and any data outside this repository are also
   backed up.
5. Confirm the documented upgrade path for the exact source and target L4T
   releases. Prepare an R39.2 installation USB or initrd-flash host as recovery.

The official quick-start command uses `--erase-all`; that option destroys the
current NVMe contents. Omitting it is not treated as a backup strategy.

## Post-upgrade gate

After completing OEM setup and installing the CUDA/TensorRT development
packages, restore this repository and run:

```bash
./scripts/probe_mig.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel "$(nproc)"
ctest --test-dir build --output-on-failure
```

`probe_mig.sh` succeeds only when the running BSP is R39.2 or newer, MIG mode is
enabled, and two MIG instances are visible. `configure_thor_mig.sh` performs the
administrator operations, creates profiles 83 and 78, starts MPS on `1g` first,
and assigns GDM to `2g`. The order is required on the tested R39.2 preview stack;
opening the graphics-capable instance first can make context creation on `1g`
hang.

## Experiment boundary

Thor supports at most two concurrent GPU instances. The primary experiment
assigns the latency-critical workload to one instance and best-effort workloads
to the other. MPS is evaluated inside an instance; it is not presented as a
replacement for MIG isolation. Cross-instance transport and residual contention
through unified LPDDR, CPU dispatch, and the power envelope are measured
separately.

Official references:

- [JetPack 7.2 downloads and release notes](https://developer.nvidia.com/embedded/jetpack/downloads)
- [Jetson Thor MIG procedure](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/MiG.html)
- [Jetson Thor flashing support](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/FlashingSupportJetsonThor.html)
