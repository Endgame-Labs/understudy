//! understudy: a CPU reflex layer in front of a systemone decision model (any System One lead, or a chat model through the adapter). hunch's statistical
//! architecture (learn from the lead, graduate on cross-validated Wilson bounds, forward novel inputs, audit a
//! sample forever, demote on bad audits) with the `deep` understudent: readouts over EmbeddingGemma 2's per-token hidden
//! states (mean+max pooling over layers, learned attention pooling, pair features). See SPEC.md.

pub mod abstain;
pub mod chat;
pub mod config;
pub mod diskcache;
pub mod checklist;
pub mod encoder;
pub mod shape;
pub mod linear;
pub mod memo;
pub mod modelfetch;
pub mod netdesign;
pub mod keywords;
pub mod ops;
pub mod snapshot;
pub mod proxy;
pub mod pyjson;
pub mod question;
pub mod readout;
pub mod replay;
pub mod rng;
pub mod rules;
pub mod serve;
pub mod simd;
pub mod stats;
pub mod understudent;
pub mod lazyenc;
pub mod lead;
