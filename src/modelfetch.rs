//! `understudy model fetch` and `understudy model status`: download the EmbeddingGemma 2 checkpoint from the
//! Hugging Face hub at a pinned revision, verify it by SHA-256, and report what a model directory holds.
//!
//! Loading never hashes (it must stay fast); only these two commands do.

use anyhow::{Context, Result, anyhow, bail, ensure};
use eg2cpu::{MODEL_FILES, MODEL_REPO, PINNED_REVISION};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// A file's expected content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expect {
    pub name: String,
    /// `None` only for a non-pinned revision's non-LFS file, which the hub gives no SHA-256 for.
    pub sha256: Option<String>,
    pub size: Option<u64>,
}

/// SHA-256 and size of the pinned revision's files. `tokenizer.json` and `model.safetensors` are LFS files, so these
/// two equal the hub's `x-linked-etag`; `config.json` is a plain git blob (the hub's etag is its git SHA-1,
/// `dc524b2cbef566d66d8533c67dc68a6a8f5d340a`), so its SHA-256 was computed from the file.
pub const PINNED: [(&str, &str, u64); 3] = [
    ("config.json", "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3", 4455),
    ("tokenizer.json", "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4", 32170510),
    ("model.safetensors", "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79", 1488915288),
];

pub fn pinned_table() -> Vec<Expect> {
    PINNED.iter().map(|(n, h, s)| Expect { name: n.to_string(), sha256: Some(h.to_string()), size: Some(*s) }).collect()
}

/// What a model file looks like on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileState {
    Ok,
    Missing,
    /// The SHA-256 it actually has.
    Mismatch(String),
    /// Present, but there is no hash to compare against.
    Unverified,
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Hash the files of `dir` against `expect`.
pub fn check_dir(dir: &Path, expect: &[Expect]) -> Result<Vec<(String, FileState)>> {
    expect
        .iter()
        .map(|e| {
            let p = dir.join(&e.name);
            let st = if !p.is_file() {
                FileState::Missing
            } else {
                match &e.sha256 {
                    None => FileState::Unverified,
                    Some(want) => {
                        let got = sha256_file(&p)?;
                        if &got == want { FileState::Ok } else { FileState::Mismatch(got) }
                    }
                }
            };
            Ok((e.name.clone(), st))
        })
        .collect()
}

// ---------------------------------------------------------------- HTTP

/// `scheme://host:port` of a URL, used to decide where the bearer token may go.
fn origin_of(u: &url::Url) -> String {
    u.origin().ascii_serialization()
}

/// A URL safe to print: no userinfo, query or fragment (signed download URLs carry credentials there).
fn redact_url(u: &str) -> String {
    match url::Url::parse(u) {
        Ok(mut p) => {
            let _ = p.set_username("");
            let _ = p.set_password(None);
            p.set_query(None);
            p.set_fragment(None);
            p.to_string()
        }
        Err(_) => u.split(['?', '#']).next().unwrap_or("").to_string(),
    }
}

/// Redact every URL inside free text (a transport error message may quote the request URL, possibly in quotes or
/// parentheses).
fn redact_text(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = ["http://", "https://"].iter().filter_map(|p| rest.find(p)).min() {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let end = tail.find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | '(' | ')' | ',')).unwrap_or(tail.len());
        out.push_str(&redact_url(&tail[..end]));
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

/// Resolve a `Location` header (absolute, scheme-relative, path-relative, ...) against the current URL.
fn resolve_redirect(current: &str, loc: &str) -> Result<String> {
    let base = url::Url::parse(current).map_err(|_| anyhow!("bad URL {}", redact_url(current)))?;
    let next = base.join(loc).map_err(|_| anyhow!("{}: bad redirect Location", redact_url(current)))?;
    Ok(next.to_string())
}

/// GET or HEAD `url`, following up to 8 redirects by hand so the bearer token only goes to the first origin.
fn http(method: &str, url: &str, token: Option<&str>, follow: bool) -> Result<ureq::Response> {
    let agent = ureq::AgentBuilder::new().redirects(0).timeout_connect(Duration::from_secs(20)).build();
    let first = origin_of(&url::Url::parse(url).map_err(|_| anyhow!("bad URL {}", redact_url(url)))?);
    let mut url = url.to_string();
    for _ in 0..8 {
        let mut req = agent.request(method, &url);
        let cur = url::Url::parse(&url).map_err(|_| anyhow!("bad URL {}", redact_url(&url)))?;
        if let Some(t) = token.filter(|_| origin_of(&cur) == first) {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        let resp = match req.call() {
            Ok(r) => r,
            Err(ureq::Error::Status(c, _)) => bail!("{method} {}: HTTP {c}", redact_url(&url)),
            Err(e) => bail!("{method} {}: {}", redact_url(&url), redact_text(e.to_string().lines().next().unwrap_or("request failed"))),
        };
        if !follow || !(300..400).contains(&resp.status()) {
            return Ok(resp);
        }
        let loc = resp.header("location").ok_or_else(|| anyhow!("{}: redirect without Location", redact_url(&url)))?;
        url = resolve_redirect(&url, loc)?;
    }
    bail!("{}: too many redirects", redact_url(&url))
}

/// The hub's SHA-256 for an LFS file (its `x-linked-etag`), if the file is stored in LFS.
pub fn remote_lfs_sha(base: &str, revision: &str, name: &str, token: Option<&str>) -> Result<Option<String>> {
    let resp = http("HEAD", &resolve_url(base, revision, name), token, false)?;
    Ok(resp
        .header("x-linked-etag")
        .map(|e| e.trim_matches('"').to_string())
        .filter(|e| e.len() == 64 && e.bytes().all(|b| b.is_ascii_hexdigit())))
}

pub fn resolve_url(base: &str, revision: &str, name: &str) -> String {
    format!("{}/{MODEL_REPO}/resolve/{revision}/{name}", base.trim_end_matches('/'))
}

fn mib(n: u64) -> f64 {
    n as f64 / (1 << 20) as f64
}

/// Create a fresh temp file next to `dest`, exclusively (`create_new` never follows a symlink or reuses a file), so a
/// pre-planted name cannot redirect the write. Retries with a new name on a collision.
fn create_temp(dest: &Path) -> Result<(File, PathBuf)> {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..100 {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let k = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dest.with_extension(format!("part{}-{nanos:x}-{k}", std::process::id()));
        match File::options().write(true).create_new(true).open(&path) {
            Ok(f) => return Ok((f, path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).with_context(|| format!("create {}", path.display())),
        }
    }
    bail!("could not create a temp file next to {}", dest.display())
}

/// Download `url` to `dest` through a temp file in the same directory: hash while streaming, check size and
/// SHA-256, and only then rename into place. A mismatch deletes the temp file and errors.
fn download(url: &str, dest: &Path, e: &Expect, token: Option<&str>) -> Result<()> {
    let (mut f, tmp) = create_temp(dest)?;
    let result = (|| -> Result<()> {
        let resp = http("GET", url, token, true)?;
        ensure!(resp.status() == 200, "GET {}: HTTP {}", redact_url(url), resp.status());
        let total = resp.header("content-length").and_then(|v| v.parse::<u64>().ok()).or(e.size);
        let mut r = resp.into_reader();
        let (mut h, mut got) = (Sha256::new(), 0u64);
        let (mut buf, start, mut last) = (vec![0u8; 1 << 20], Instant::now(), Instant::now());
        loop {
            let n = r.read(&mut buf).with_context(|| format!("reading {}", e.name))?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
            f.write_all(&buf[..n])?;
            got += n as u64;
            if last.elapsed() > Duration::from_millis(250) {
                last = Instant::now();
                match total {
                    Some(t) => eprint!("\r  {} {:.0}/{:.0} MiB ({:.0}%)   ", e.name, mib(got), mib(t), 100.0 * got as f64 / t as f64),
                    None => eprint!("\r  {} {:.0} MiB   ", e.name, mib(got)),
                }
            }
        }
        f.sync_all()?;
        eprintln!("\r  {} {:.1} MiB in {:.1}s{}", e.name, mib(got), start.elapsed().as_secs_f64(), " ".repeat(16));
        if let Some(want) = e.size {
            ensure!(got == want, "{}: got {got} bytes, expected {want}", e.name);
        }
        let sum = hex(&h.finalize());
        match &e.sha256 {
            Some(want) => ensure!(&sum == want, "{}: SHA-256 mismatch (got {sum}, expected {want})", e.name),
            None => eprintln!("  warning: {} has no known SHA-256 for this revision; unverified (sha256 {sum})", e.name),
        }
        std::fs::rename(&tmp, dest)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Make sure every file of `expect` is in `dir` with the right hash, downloading what is missing or wrong.
pub fn fetch_files(dir: &Path, base: &str, revision: &str, expect: &[Expect], token: Option<&str>) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for e in expect {
        let dest = dir.join(&e.name);
        if let (true, Some(want)) = (dest.is_file(), &e.sha256) {
            if &sha256_file(&dest)? == want {
                eprintln!("  {} already present, hash matches", e.name);
                continue;
            }
            eprintln!("  {} is present with a different hash; downloading it again", e.name);
        }
        download(&resolve_url(base, revision, &e.name), &dest, e, token)?;
    }
    Ok(())
}

// ---------------------------------------------------------------- commands

fn is_revision(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The expectation for a file of a non-pinned revision. Only `config.json` (a plain git blob, no LFS hash) may go
/// unverified; the LFS files must have a valid hash from the hub, or the fetch fails rather than install unchecked.
fn lfs_expect(name: &str, sha: Option<String>) -> Result<Expect> {
    if sha.is_none() && name != "config.json" {
        bail!("{name}: the host gave no valid SHA-256 (x-linked-etag) for it, so it cannot be verified; refusing to install it");
    }
    Ok(Expect { name: name.to_string(), sha256: sha, size: None })
}

/// The expected files of a non-pinned revision, with the hub's SHA-256 for each LFS file.
pub fn alt_expect(base: &str, rev: &str, token: Option<&str>) -> Result<Vec<Expect>> {
    MODEL_FILES.iter().map(|n| lfs_expect(n, remote_lfs_sha(base, rev, n, token)?)).collect()
}

/// `understudy model fetch`. Returns the directory holding the model.
pub fn fetch(revision: Option<&str>, dir: Option<&Path>) -> Result<PathBuf> {
    let rev = revision.unwrap_or(PINNED_REVISION);
    ensure!(is_revision(rev), "--revision takes a full 40-character commit hash, not {rev:?}");
    let base = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    let token = token.as_deref();
    let hub = eg2cpu::hub_cache_dir();
    let target = match dir {
        Some(d) => d.to_path_buf(),
        None => eg2cpu::snapshots_dir(&hub).join(rev),
    };
    let expect = if rev == PINNED_REVISION {
        pinned_table()
    } else {
        eprintln!("revision {rev} is not the pinned {PINNED_REVISION}; checking LFS files against the hub's SHA-256");
        alt_expect(&base, rev, token)?
    };
    eprintln!("fetching {MODEL_REPO}@{rev} into {}", target.display());
    let t0 = Instant::now();
    fetch_files(&target, &base, rev, &expect, token)?;
    if dir.is_none() && rev == PINNED_REVISION {
        let refs = target.parent().and_then(Path::parent).map(|r| r.join("refs"));
        if let Some(refs) = refs {
            std::fs::create_dir_all(&refs)?;
            std::fs::write(refs.join("main"), rev)?;
        }
    }
    eprintln!("done in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(target)
}

/// `understudy model status`: resolve the directory the loader would use and hash its files against the pins.
/// Returns true when everything matches.
pub fn status() -> Result<bool> {
    let (dir, revision) = if let Ok(d) = std::env::var("EG2_MODEL_DIR") {
        (PathBuf::from(&d), "unknown (EG2_MODEL_DIR)".to_string())
    } else {
        let snaps = eg2cpu::snapshots_dir(&eg2cpu::hub_cache_dir());
        match eg2cpu::select_snapshot(&snaps, PINNED_REVISION) {
            Some(s) => (s.dir, if s.pinned { format!("{} (pinned)", s.revision) } else { format!("{} (NOT the pinned {PINNED_REVISION})", s.revision) }),
            None => {
                println!("model dir: none (looked in {})", snaps.display());
                println!("run `understudy model fetch`, or set EG2_MODEL_DIR");
                return Ok(false);
            }
        }
    };
    println!("model dir: {}", dir.display());
    println!("revision:  {revision}");
    println!("pinned:    {PINNED_REVISION}");
    eprintln!("hashing the files (takes a few seconds)...");
    let mut all = true;
    for (name, st) in check_dir(&dir, &pinned_table())? {
        let msg = match &st {
            FileState::Ok => "ok".to_string(),
            FileState::Missing => "MISSING".to_string(),
            FileState::Mismatch(got) => format!("MISMATCH (sha256 {got})"),
            FileState::Unverified => "unverified".to_string(),
        };
        all &= st == FileState::Ok;
        println!("{name:<18} {msg}");
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("understudy-modelfetch-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn sha(b: &[u8]) -> String {
        hex(&Sha256::digest(b))
    }

    fn exp(name: &str, body: &[u8]) -> Expect {
        Expect { name: name.into(), sha256: Some(sha(body)), size: Some(body.len() as u64) }
    }

    #[test]
    fn pinned_table_is_well_formed() {
        for (n, h, _) in PINNED {
            assert!(MODEL_FILES.contains(&n));
            assert!(h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        assert!(is_revision(PINNED_REVISION));
    }

    #[test]
    fn check_dir_reports_ok_missing_mismatch_unverified() {
        let d = tmp("check");
        std::fs::write(d.join("a"), b"hello").unwrap();
        std::fs::write(d.join("b"), b"tampered").unwrap();
        std::fs::write(d.join("d"), b"anything").unwrap();
        let table = vec![
            exp("a", b"hello"),
            exp("b", b"original"),
            exp("c", b"never written"),
            Expect { name: "d".into(), sha256: None, size: None },
        ];
        let got = check_dir(&d, &table).unwrap();
        assert_eq!(got[0].1, FileState::Ok);
        assert_eq!(got[1].1, FileState::Mismatch(sha(b"tampered")));
        assert_eq!(got[2].1, FileState::Missing);
        assert_eq!(got[3].1, FileState::Unverified);
    }

    #[test]
    fn sha256_of_known_input() {
        let d = tmp("sha");
        std::fs::write(d.join("x"), b"abc").unwrap();
        assert_eq!(sha256_file(&d.join("x")).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    /// Serve `files` (path suffix -> body) from loopback for exactly `n` requests.
    fn serve(files: Vec<(&'static str, Vec<u8>)>, n: usize) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for _ in 0..n {
                let (mut s, _) = l.accept().unwrap();
                let mut line = String::new();
                let mut r = BufReader::new(s.try_clone().unwrap());
                r.read_line(&mut line).unwrap();
                loop {
                    let mut h = String::new();
                    r.read_line(&mut h).unwrap();
                    if h.trim().is_empty() {
                        break;
                    }
                }
                let path = line.split_whitespace().nth(1).unwrap_or("");
                match files.iter().find(|(p, _)| path.ends_with(p)) {
                    Some((_, b)) => {
                        write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", b.len()).unwrap();
                        s.write_all(b).unwrap();
                    }
                    None => write!(s, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap(),
                }
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn fetch_files_downloads_verifies_and_skips() {
        let d = tmp("fetch");
        let base = serve(vec![("/config.json", b"{}".to_vec()), ("/weights", b"WEIGHTS".to_vec())], 2);
        let table = vec![exp("config.json", b"{}"), exp("weights", b"WEIGHTS")];
        fetch_files(&d, &base, PINNED_REVISION, &table, None).unwrap();
        assert_eq!(std::fs::read(d.join("weights")).unwrap(), b"WEIGHTS");
        // A second run finds both files valid and makes no request (the server only answers 2).
        fetch_files(&d, "http://127.0.0.1:1", PINNED_REVISION, &table, None).unwrap();
        assert!(std::fs::read_dir(&d).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().contains("part")));
    }

    #[test]
    fn hash_mismatch_deletes_the_file_and_errors() {
        let d = tmp("mismatch");
        let base = serve(vec![("/weights", b"CORRUPT".to_vec())], 1);
        let table = vec![Expect { name: "weights".into(), sha256: Some(sha(b"WEIGHTS")), size: None }];
        let err = fetch_files(&d, &base, PINNED_REVISION, &table, None).unwrap_err().to_string();
        assert!(err.contains("SHA-256 mismatch"), "{err}");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0, "no partial or bad file may remain");
    }

    #[test]
    fn bad_existing_file_is_replaced() {
        let d = tmp("replace");
        std::fs::write(d.join("weights"), b"stale").unwrap();
        let base = serve(vec![("/weights", b"WEIGHTS".to_vec())], 1);
        fetch_files(&d, &base, PINNED_REVISION, &[exp("weights", b"WEIGHTS")], None).unwrap();
        assert_eq!(std::fs::read(d.join("weights")).unwrap(), b"WEIGHTS");
    }

    #[test]
    fn temp_file_is_exclusive_and_ignores_planted_symlinks() {
        let d = tmp("temp");
        let victim = d.join("victim");
        std::fs::write(&victim, b"precious").unwrap();
        let dest = d.join("weights");
        // Plant symlinks at the old predictable name and at a few likely ones.
        for suffix in [format!("part{}", std::process::id())] {
            std::os::unix::fs::symlink(&victim, dest.with_extension(suffix)).unwrap();
        }
        let (mut f, p1) = create_temp(&dest).unwrap();
        let (_f2, p2) = create_temp(&dest).unwrap();
        assert_ne!(p1, p2);
        f.write_all(b"x").unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        // create_new refuses an existing symlink outright, even a dangling one.
        let link = d.join("link");
        std::os::unix::fs::symlink(&victim, &link).unwrap();
        assert!(File::options().write(true).create_new(true).open(&link).is_err());
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
    }

    #[test]
    fn urls_are_redacted_in_errors() {
        let u = "https://user:pw@cdn.example/x/y?X-Amz-Signature=SECRET&token=T#frag";
        let r = redact_url(u);
        assert_eq!(r, "https://cdn.example/x/y");
        let t = redact_text(&format!("Connection Failed: connect error for {u} (timed out)"));
        assert!(!t.contains("SECRET") && !t.contains("pw") && t.contains("https://cdn.example/x/y"), "{t}");
        assert!(!redact_url("not a url?secret=1").contains("secret"));
    }

    #[test]
    fn failed_request_error_hides_the_query() {
        let err = http("GET", "http://127.0.0.1:1/f?sig=SECRET", None, true).unwrap_err().to_string();
        assert!(!err.contains("SECRET"), "{err}");
    }

    #[test]
    fn redirects_resolve_like_urls() {
        let cur = "https://hub.example/a/b/file?x=1";
        assert_eq!(resolve_redirect(cur, "weights").unwrap(), "https://hub.example/a/b/weights");
        assert_eq!(resolve_redirect(cur, "/root/w?sig=1").unwrap(), "https://hub.example/root/w?sig=1");
        assert_eq!(resolve_redirect(cur, "//cdn.example/f").unwrap(), "https://cdn.example/f");
        assert_eq!(resolve_redirect(cur, "http://other.example:8080/f").unwrap(), "http://other.example:8080/f");
    }

    /// A loopback server that answers each request with `reply(path)` and records (path, Authorization header).
    fn serve_log(n: usize, reply: impl Fn(&str) -> String + Send + 'static) -> (String, std::sync::Arc<std::sync::Mutex<Vec<(String, Option<String>)>>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log2 = log.clone();
        std::thread::spawn(move || {
            for _ in 0..n {
                let (mut s, _) = l.accept().unwrap();
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                let mut auth = None;
                loop {
                    let mut h = String::new();
                    r.read_line(&mut h).unwrap();
                    if h.trim().is_empty() {
                        break;
                    }
                    if h.to_ascii_lowercase().starts_with("authorization:") {
                        auth = Some(h.split_once(':').unwrap().1.trim().to_string());
                    }
                }
                let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                // Record before responding, so a client that has its response can always see the request in the log.
                let response = reply(&path);
                log2.lock().unwrap().push((path, auth));
                s.write_all(response.as_bytes()).unwrap();
            }
        });
        (format!("http://{addr}"), log)
    }

    #[test]
    fn relative_redirect_is_followed_and_token_stays_on_the_origin() {
        let (base, log) = serve_log(2, |p| match p {
            "/start" => "HTTP/1.1 302 Found\r\nLocation: next\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(),
            _ => "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".into(),
        });
        let r = http("GET", &format!("{base}/start"), Some("TOK"), true).unwrap();
        assert_eq!(r.status(), 200);
        let log = log.lock().unwrap();
        assert_eq!(log[1].0, "/next");
        assert!(log.iter().all(|(_, a)| a.as_deref() == Some("Bearer TOK")));
    }

    #[test]
    fn cross_origin_redirect_drops_the_token() {
        let (other, olog) = serve_log(1, |_| "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".into());
        let target = format!("{other}/final");
        let (base, _) = serve_log(1, move |_| format!("HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"));
        let r = http("GET", &format!("{base}/start"), Some("TOK"), true).unwrap();
        assert_eq!(r.status(), 200);
        let olog = olog.lock().unwrap();
        assert_eq!(olog[0], ("/final".to_string(), None), "token must not reach another origin");
    }

    #[test]
    fn unverifiable_lfs_file_is_refused_but_config_may_be_unverified() {
        assert!(lfs_expect("model.safetensors", None).unwrap_err().to_string().contains("cannot be verified"));
        assert!(lfs_expect("tokenizer.json", None).is_err());
        assert_eq!(lfs_expect("model.safetensors", Some("ab".repeat(32))).unwrap().sha256, Some("ab".repeat(32)));
        assert!(lfs_expect("config.json", None).unwrap().sha256.is_none());
    }

    #[test]
    fn remote_lfs_sha_rejects_a_missing_or_malformed_header() {
        let (base, _) = serve_log(2, |p| {
            let etag = if p.ends_with("bad") { "x-linked-etag: \"nothex\"\r\n" } else { "" };
            format!("HTTP/1.1 200 OK\r\n{etag}Content-Length: 0\r\nConnection: close\r\n\r\n")
        });
        assert_eq!(remote_lfs_sha(&base, PINNED_REVISION, "bad", None).unwrap(), None);
        assert_eq!(remote_lfs_sha(&base, PINNED_REVISION, "none", None).unwrap(), None);
    }

    #[test]
    fn revision_must_be_a_full_hash() {
        assert!(fetch(Some("main"), None).unwrap_err().to_string().contains("40-character"));
    }

    // ---- raw test server: `handler(request head) -> full raw response`, for exactly `n` requests.
    fn serve_raw(n: usize, handler: impl Fn(&str) -> Vec<u8> + Send + 'static) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for _ in 0..n {
                let (mut s, _) = l.accept().unwrap();
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut head = String::new();
                loop {
                    let mut h = String::new();
                    r.read_line(&mut h).unwrap();
                    if h.trim().is_empty() {
                        break;
                    }
                    head.push_str(&h);
                }
                s.write_all(&handler(&head)).unwrap();
            }
        });
        format!("http://{addr}")
    }

    fn ok(body: &str) -> Vec<u8> {
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).into_bytes()
    }

    fn redirect(loc: &str) -> Vec<u8> {
        format!("HTTP/1.1 302 Found\r\nLocation: {loc}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").into_bytes()
    }

    fn has_auth(head: &str) -> bool {
        head.to_ascii_lowercase().contains("authorization: bearer")
    }

    fn body_of(r: ureq::Response) -> String {
        let mut b = String::new();
        r.into_reader().read_to_string(&mut b).unwrap();
        b
    }

    #[cfg(unix)]
    #[test]
    fn planted_symlink_is_not_written_through() {
        let d = tmp("symlink");
        let victim = d.join("victim");
        std::fs::write(&victim, b"precious").unwrap();
        let out = d.join("out");
        std::fs::create_dir_all(&out).unwrap();
        // Plant the old predictable temp name.
        std::os::unix::fs::symlink(&victim, out.join(format!("weights.part{}", std::process::id()))).unwrap();
        let base = serve(vec![("/weights", b"WEIGHTS".to_vec())], 1);
        fetch_files(&out, &base, PINNED_REVISION, &[exp("weights", b"WEIGHTS")], None).unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        assert_eq!(std::fs::read(out.join("weights")).unwrap(), b"WEIGHTS");
        // create_temp itself refuses an existing name and picks another.
        let (_f, p) = create_temp(&out.join("weights")).unwrap();
        assert!(!p.is_symlink());
    }

    #[test]
    fn errors_redact_url_credentials() {
        assert_eq!(redact_url("https://u:p@h.example/a/b?X-Amz-Signature=s3cr3t&t=1#frag"), "https://h.example/a/b");
        assert_eq!(redact_text("failed (https://h.example/a?sig=s3cr3t): refused"), "failed (https://h.example/a): refused");
        // A connection failure to a signed URL, via a redirect from a first host.
        let dead = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let dead_addr = dead.local_addr().unwrap();
        drop(dead);
        let loc = format!("http://{dead_addr}/f?X-Amz-Signature=s3cr3t&token=tok3n");
        let base = serve_raw(1, move |_| redirect(&loc));
        let err = http("GET", &format!("{base}/start?sig=first"), None, true).unwrap_err().to_string();
        assert!(err.contains(&dead_addr.to_string()), "{err}");
        for secret in ["s3cr3t", "tok3n", "X-Amz", "sig=first"] {
            assert!(!err.contains(secret), "{err}");
        }
        // HTTP status errors too.
        let base = serve_raw(1, |_| b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec());
        let err = http("GET", &format!("{base}/x?sig=s3cr3t"), None, true).unwrap_err().to_string();
        assert!(err.contains("403") && !err.contains("s3cr3t"), "{err}");
    }

    #[test]
    fn alternate_revision_requires_lfs_hashes() {
        let good = "ab".repeat(32);
        let head = |etag: Option<String>| {
            move |req: &str| {
                let lfs = !req.contains("config.json");
                let mut r = String::from("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n");
                if let (true, Some(e)) = (lfs, &etag) {
                    r += &format!("x-linked-etag: \"{e}\"\r\n");
                }
                (r + "\r\n").into_bytes()
            }
        };
        let base = serve_raw(3, head(Some(good.clone())));
        let t = alt_expect(&base, PINNED_REVISION, None).unwrap();
        assert_eq!(t[0].sha256, None);
        assert_eq!(t[1].sha256.as_deref(), Some(good.as_str()));
        assert_eq!(t[2].sha256.as_deref(), Some(good.as_str()));
        // Missing header: refuse (config.json is first and passes; the next file fails).
        let base = serve_raw(2, head(None));
        let err = alt_expect(&base, PINNED_REVISION, None).unwrap_err().to_string();
        assert!(err.contains("tokenizer.json") && err.contains("x-linked-etag"), "{err}");
        // Malformed header: refuse.
        let base = serve_raw(2, head(Some("not-a-sha".into())));
        let err = alt_expect(&base, PINNED_REVISION, None).unwrap_err().to_string();
        assert!(err.contains("tokenizer.json"), "{err}");
    }

    #[test]
    fn redirects_resolve_as_url_references_and_scope_the_token() {
        // Path-relative: `Location: weights` from /a/b/start -> /a/b/weights, same origin, token kept.
        let base = serve_raw(2, |req| {
            if req.starts_with("GET /a/b/weights ") {
                assert!(has_auth(req));
                ok("REL")
            } else {
                redirect("weights")
            }
        });
        assert_eq!(body_of(http("GET", &format!("{base}/a/b/start"), Some("t"), true).unwrap()), "REL");

        // Root-relative.
        let base = serve_raw(2, |req| if req.starts_with("GET /root ") { assert!(has_auth(req)); ok("ROOT") } else { redirect("/root") });
        assert_eq!(body_of(http("GET", &format!("{base}/a/start"), Some("t"), true).unwrap()), "ROOT");

        // Scheme-relative to another origin: goes there (not to the first host), without the token.
        let other = serve_raw(1, |req| {
            assert!(!has_auth(req), "token leaked to another origin: {req}");
            ok("CDN")
        });
        let other_host = other.trim_start_matches("http:").to_string(); // //127.0.0.1:port
        let base = serve_raw(1, move |_| redirect(&format!("{other_host}/file")));
        assert_eq!(body_of(http("GET", &format!("{base}/start"), Some("t"), true).unwrap()), "CDN");

        // Absolute cross-origin: same, without the token.
        let other = serve_raw(1, |req| {
            assert!(!has_auth(req), "token leaked to another origin: {req}");
            ok("ABS")
        });
        let base = serve_raw(1, move |_| redirect(&format!("{other}/file")));
        assert_eq!(body_of(http("GET", &format!("{base}/start"), Some("t"), true).unwrap()), "ABS");
    }

    /// Real network: the pins match the hub's LFS hashes, and a fetch into a temp dir verifies. Downloads ~1.5 GB.
    /// Run with `cargo test --release -- --ignored model_fetch`.
    #[test]
    #[ignore]
    fn model_fetch_into_a_temp_dir() {
        let d = tmp("real");
        let base = "https://huggingface.co";
        for (n, h, _) in PINNED {
            if let Some(remote) = remote_lfs_sha(base, PINNED_REVISION, n, None).unwrap() {
                assert_eq!(remote, h, "{n}: pin differs from the hub's x-linked-etag");
            }
        }
        fetch_files(&d, base, PINNED_REVISION, &pinned_table(), None).unwrap();
        assert!(check_dir(&d, &pinned_table()).unwrap().iter().all(|(_, s)| *s == FileState::Ok));
        let _ = std::fs::remove_dir_all(&d);
    }
}
