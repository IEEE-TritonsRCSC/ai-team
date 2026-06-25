# Installing the patched `rcssserver` (catch-glue dribble + native `drop`)

This makes the **patched** rcssserver your default `rcssserver` on `PATH`, so running with
`--env sim-only` directly uses our dribble changes (catch-glue ball carry + native `drop`) —
no extra flags needed.

You need two things from the repo (in `robocup_downloads/`):
- `rcssserver-19.0.0.tar.gz` — the pristine, version-pinned source
- `ssim_patches/server.patch` and `ssim_patches/server_macos.patch` — our changes

> **Always patch a *fresh* extraction of the tarball.** Never patch a tree you've already
> modified — that's what breaks when everyone's sim is in a different state.

---

## 1. Install build dependencies (once)

**Ubuntu / Debian (and WSL2):**
```bash
sudo apt update
sudo apt install -y build-essential libboost-all-dev flex bison \
                    automake autoconf libtool autoconf-archive
```

**macOS (Homebrew):**
```bash
brew install boost flex bison automake autoconf libtool autoconf-archive
# Homebrew's bison/flex are keg-only — put them ahead of the ancient system ones:
export PATH="$(brew --prefix bison)/bin:$(brew --prefix flex)/bin:$PATH"
```

**Windows:** there is no native build. Install **WSL2 + Ubuntu**, then follow the Ubuntu
steps inside WSL:
```powershell
wsl --install -d Ubuntu      # run in PowerShell as Admin, then reboot
```
Open the **Ubuntu** terminal and do everything below there. On **Windows 11** the monitor
GUI just works (WSLg). On **Windows 10** install an X server (e.g. VcXsrv) and
`export DISPLAY=:0` before launching the monitor.

---

## 2. Patch, build, install

Run from the directory that contains `rcssserver-19.0.0.tar.gz` and `ssim_patches/`
(i.e. `robocup_downloads/`):

```bash
tar xzf rcssserver-19.0.0.tar.gz                 # pristine, version-pinned base
patch -p0 < ssim_patches/server.patch            # catch-glue + native drop (all OSes)
patch -p0 < ssim_patches/server_macos.patch      # modern Boost/autoconf fix (all OSes)

cd rcssserver-19.0.0
autoreconf -i                                    # regenerate ./configure from patched configure.ac
./configure
make -j4                                          # regenerates the parser from the patched .ypp/.lpp
sudo make install                                 # installs to /usr/local/bin, replacing the old rcssserver
```

---

## 3. Verify

```bash
which rcssserver            # -> /usr/local/bin/rcssserver
ls -l $(which rcssserver)   # mtime should be "just now", not an old date
```

After this, the default works with no `--sim-cmd`:
```bash
python launch_infer.py <checkpoint> --env sim-only ...
```
and the dribble/carry + `drop` behavior shows up in the monitor.

---

## Why these non-obvious steps matter

- **`autoreconf -i` is required** because we patched `configure.ac`; the tarball's shipped
  `configure` was generated from the *old* one and would silently ignore the change.
  (That's why `autoconf-archive` is a dependency — it provides the `AX_BOOST_BASE` macro.)
- **bison + flex are required** because the patch ships the grammar *source* (`.ypp`/`.lpp`),
  not the generated parser. `make` regenerates it; without them the new `drop` command
  won't build.
- **`sudo make install`** overwrites your existing `/usr/local/bin/rcssserver` — that's the
  goal: PATH now points at the patched build.

---

## Troubleshooting

- **`configure` fails on Boost.System / `AX_BOOST_SYSTEM`** → you skipped
  `server_macos.patch`. Apply it and re-run `autoreconf -i && ./configure`.
- **`autoreconf: command not found` / `AX_BOOST_BASE ... not found`** → missing autotools;
  install `automake autoconf libtool autoconf-archive` (see step 1).
- **Parser/`drop` build errors / "syntax error near drop"** → bison or flex missing (macOS:
  also make sure the Homebrew `PATH` export above is in effect, not the old system bison).
- **Old Boost (<1.81)** → in that rare case skip `server_macos.patch` and use the stock
  `./configure` without `autoreconf` (the Boost.System skip would break linking on old Boost).
- **Monitor doesn't open in WSL** → Windows 11 needs nothing; Windows 10 needs an X server
  running and `export DISPLAY=:0`.
