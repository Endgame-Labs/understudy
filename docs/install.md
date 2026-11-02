# Install

Understudy ships as a single `understudy` binary. Releases are built for four targets: `x86_64-unknown-linux-gnu`,
`aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`, and `aarch64-apple-darwin`. The current release is
`v0.1.0-preview.2`, a GitHub pre-release.

| channel | command |
|---|---|
| Homebrew | `brew install endgame-labs/oss/understudy` |
| Shell installer | `curl -fsSL https://github.com/endgame-labs/understudy/releases/latest/download/understudy-installer.sh \| sh` |
| Debian package | `gh release download v0.1.0-preview.2 --repo endgame-labs/understudy --pattern '*_amd64.deb' && sudo apt install ./understudy_*_amd64.deb` |
| Container image | `docker run -p 8787:8787 -v understudy-data:/data ghcr.io/endgame-labs/understudy:preview --lead-url <URL>` |
| Source | `cargo install --git https://github.com/endgame-labs/understudy` |

## Homebrew

```bash
brew install endgame-labs/oss/understudy
```

The formula lives in the [Endgame-Labs/homebrew-oss](https://github.com/Endgame-Labs/homebrew-oss) tap and is updated on every
release, pre-releases included. Homebrew verifies the archive's SHA-256 from the formula.

## Shell installer

GitHub's `releases/latest` URL skips pre-releases. Until a full (non-pre-release) release exists, it returns 404, so pin the
version:

```bash
curl -fsSL https://github.com/endgame-labs/understudy/releases/download/v0.1.0-preview.2/understudy-installer.sh | sh
```

Once a full release exists, the unpinned form installs the newest one:

```bash
curl -fsSL https://github.com/endgame-labs/understudy/releases/latest/download/understudy-installer.sh | sh
```

The installer downloads the archive for the host platform, verifies its checksum, and places `understudy` in
`$CARGO_HOME/bin` (default `~/.cargo/bin`).

## Debian package

Each release carries an `amd64` and an `arm64` `.deb` as release assets. Download the one for the host and install it with
`apt`, which resolves dependencies. GitHub can rewrite the `~` in an asset name (for example to `.`), so the commands match
the architecture suffix instead of the exact name:

```bash
gh release download v0.1.0-preview.2 --repo endgame-labs/understudy --pattern '*_amd64.deb'
sudo apt install ./understudy_*_amd64.deb
```

Use `*_arm64.deb` on arm64 hosts. The exact asset names are listed on the release page.

In Debian version syntax, `~` sorts a pre-release before the release, so `0.1.0~preview.2` is older than `0.1.0`. There is no
hosted apt repository yet.

## Container image

```bash
docker run -p 8787:8787 -v understudy-data:/data \
  ghcr.io/endgame-labs/understudy:preview --lead-url <URL>
```

The entrypoint runs `understudy serve`, so extra arguments go to `serve`. To check the version, override the entrypoint:

```bash
docker run --rm --entrypoint understudy ghcr.io/endgame-labs/understudy:preview --version
```

| item | value |
|---|---|
| Port | `8787` (the HTTP listener) |
| Volume | `/data` (snapshots and state; mount a named volume or a host directory so a restart keeps what was learned) |
| Arguments | the same as `understudy serve`; see the [API](api.md#modes-and-serve-options) |
| Tags | `preview` follows the latest pre-release; versioned tags match release tags |

The listener has no authentication. Publishing the port exposes every request, and the lead credentials Understudy forwards, to
anything that can reach it. Bind it to a private interface (for example `-p 127.0.0.1:8787:8787`) or put it behind an
authenticating proxy. See [Security](security.md#listener).

## From source

```bash
cargo install --git https://github.com/endgame-labs/understudy
```

This needs a Rust toolchain at the version in `rust-version` in `Cargo.toml`. The crate is not on crates.io: the name
`understudy` is held by an unrelated 0.0.0 placeholder.

## Checksums

Every release archive has a `.sha256` file next to it, and the release also carries `sha256.sum` with all of them.

```bash
curl -fsSLO https://github.com/endgame-labs/understudy/releases/download/v0.1.0-preview.2/understudy-x86_64-unknown-linux-gnu.tar.xz
curl -fsSLO https://github.com/endgame-labs/understudy/releases/download/v0.1.0-preview.2/understudy-x86_64-unknown-linux-gnu.tar.xz.sha256
sha256sum -c understudy-x86_64-unknown-linux-gnu.tar.xz.sha256
```

On macOS, use `shasum -a 256 -c` in place of `sha256sum -c`.

## Verify

```bash
understudy --version
```

The output is `understudy 0.1.0-preview.2`.
