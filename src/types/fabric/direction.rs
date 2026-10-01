//! One-way asymmetry for NICs and inter-node links.
//!
//! A NIC may transmit (egress) or receive (ingress) slower than its line
//! rate; a custom inter-node link may carry different bandwidth or latency
//! in each direction. Symmetric hardware needs none of this, so both are
//! optional and absent by default.

use std::fmt::{Display, Formatter};

use crate::types::common::{Bandwidth, Latency, NodeId};

/// Per-direction caps on a NIC's line rate. The effective rate in a
/// direction is `min(line rate, cap)`. At least one direction is capped.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NicDirectionCaps {
    Egress(Bandwidth),
    Ingress(Bandwidth),
    Both {
        egress: Bandwidth,
        ingress: Bandwidth,
    },
}

impl NicDirectionCaps {
    /// `None` when neither direction is capped (a symmetric NIC).
    pub fn new(egress: Option<Bandwidth>, ingress: Option<Bandwidth>) -> Option<Self> {
        match (egress, ingress) {
            (None, None) => None,
            (Some(egress), None) => Some(Self::Egress(egress)),
            (None, Some(ingress)) => Some(Self::Ingress(ingress)),
            (Some(egress), Some(ingress)) => Some(Self::Both { egress, ingress }),
        }
    }

    pub fn egress(self) -> Option<Bandwidth> {
        match self {
            Self::Egress(egress) | Self::Both { egress, .. } => Some(egress),
            Self::Ingress(_) => None,
        }
    }

    pub fn ingress(self) -> Option<Bandwidth> {
        match self {
            Self::Ingress(ingress) | Self::Both { ingress, .. } => Some(ingress),
            Self::Egress(_) => None,
        }
    }

    /// Both caps multiplied by `scale` (scenario NIC degradation).
    pub fn scaled(self, scale: f64) -> Self {
        match self {
            Self::Egress(egress) => Self::Egress(egress * scale),
            Self::Ingress(ingress) => Self::Ingress(ingress * scale),
            Self::Both { egress, ingress } => Self::Both {
                egress: egress * scale,
                ingress: ingress * scale,
            },
        }
    }
}

/// Bandwidth and latency of one direction of a link.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DirectionProfile {
    pub bandwidth: Bandwidth,
    pub latency: Latency,
}

/// A link whose `from_node -> to_node` (forward) and reverse directions
/// differ. Both directions are fully resolved at parse time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AsymmetricLink {
    from_node: NodeId,
    to_node: NodeId,
    forward: DirectionProfile,
    reverse: DirectionProfile,
}

impl AsymmetricLink {
    pub fn new(
        from_node: NodeId,
        to_node: NodeId,
        forward: DirectionProfile,
        reverse: DirectionProfile,
    ) -> Result<Self, DirectionError> {
        if from_node == to_node {
            return Err(DirectionError::SelfLink { node: from_node });
        }
        for (direction, profile) in [("forward", forward), ("reverse", reverse)] {
            let bandwidth = profile.bandwidth.as_bytes_per_sec();
            let latency = profile.latency.to_us();
            if !bandwidth.is_finite() || bandwidth <= 0.0 {
                return Err(DirectionError::InvalidBandwidth { direction });
            }
            if !latency.is_finite() || latency < 0.0 {
                return Err(DirectionError::InvalidLatency { direction });
            }
        }
        Ok(Self {
            from_node,
            to_node,
            forward,
            reverse,
        })
    }

    pub fn from_node(self) -> NodeId {
        self.from_node
    }

    pub fn to_node(self) -> NodeId {
        self.to_node
    }

    pub fn forward(self) -> DirectionProfile {
        self.forward
    }

    pub fn reverse(self) -> DirectionProfile {
        self.reverse
    }

    /// The profile for traffic `src -> dst`, or `None` when the pair is not
    /// this link's endpoints.
    pub fn direction(self, src: NodeId, dst: NodeId) -> Option<DirectionProfile> {
        if src == self.from_node && dst == self.to_node {
            Some(self.forward)
        } else if src == self.to_node && dst == self.from_node {
            Some(self.reverse)
        } else {
            None
        }
    }

    pub fn scaled(self, bandwidth_scale: f64, latency_scale: f64) -> Self {
        let scale = |profile: DirectionProfile| DirectionProfile {
            bandwidth: profile.bandwidth * bandwidth_scale,
            latency: Latency::from_us(profile.latency.to_us() * latency_scale),
        };
        Self {
            forward: scale(self.forward),
            reverse: scale(self.reverse),
            ..self
        }
    }
}

/// Whether a custom inter-node link behaves the same in both directions.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum LinkDirectionality {
    #[default]
    Symmetric,
    Asymmetric(AsymmetricLink),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DirectionError {
    SelfLink { node: NodeId },
    InvalidBandwidth { direction: &'static str },
    InvalidLatency { direction: &'static str },
}

impl Display for DirectionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfLink { node } => {
                write!(
                    f,
                    "a directional link needs two distinct nodes, got {node} twice"
                )
            }
            Self::InvalidBandwidth { direction } => {
                write!(f, "{direction} bandwidth must be finite and positive")
            }
            Self::InvalidLatency { direction } => {
                write!(f, "{direction} latency must be finite and non-negative")
            }
        }
    }
}

impl std::error::Error for DirectionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn gbps(value: f64) -> Bandwidth {
        Bandwidth::from_gigabits_per_sec(value)
    }

    fn profile(bandwidth_gbps: f64, latency_us: f64) -> DirectionProfile {
        DirectionProfile {
            bandwidth: gbps(bandwidth_gbps),
            latency: Latency::from_us(latency_us),
        }
    }

    #[test]
    fn nic_caps_need_at_least_one_direction() {
        assert_eq!(NicDirectionCaps::new(None, None), None);
        let egress = NicDirectionCaps::new(Some(gbps(3.61)), None).expect("caps");
        assert_eq!(egress.egress(), Some(gbps(3.61)));
        assert_eq!(egress.ingress(), None);
        let both = NicDirectionCaps::new(Some(gbps(1.0)), Some(gbps(2.0))).expect("caps");
        assert_eq!(both.ingress(), Some(gbps(2.0)));
        assert_eq!(both.scaled(0.5).egress(), Some(gbps(0.5)));
    }

    #[test]
    fn asymmetric_link_resolves_direction_by_endpoints() {
        let link =
            AsymmetricLink::new(0, 1, profile(3.61, 10.0), profile(9.41, 5.0)).expect("link");
        assert_eq!(link.direction(0, 1), Some(profile(3.61, 10.0)));
        assert_eq!(link.direction(1, 0), Some(profile(9.41, 5.0)));
        assert_eq!(link.direction(0, 2), None);
        let scaled = link.scaled(0.5, 2.0).direction(1, 0).expect("reverse");
        assert!((scaled.bandwidth.as_gigabits_per_sec() - 4.705).abs() < 1e-9);
        assert!((scaled.latency.to_us() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn asymmetric_link_rejects_invalid_profiles() {
        assert_eq!(
            AsymmetricLink::new(2, 2, profile(1.0, 1.0), profile(1.0, 1.0)),
            Err(DirectionError::SelfLink { node: 2 })
        );
        assert_eq!(
            AsymmetricLink::new(0, 1, profile(0.0, 1.0), profile(1.0, 1.0)),
            Err(DirectionError::InvalidBandwidth {
                direction: "forward"
            })
        );
        assert_eq!(
            AsymmetricLink::new(0, 1, profile(1.0, 1.0), profile(1.0, -1.0)),
            Err(DirectionError::InvalidLatency {
                direction: "reverse"
            })
        );
    }
}
