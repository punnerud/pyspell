//! SPKI leaf-key pinning `TlsVerifier` for embedded-tls.
//!
//! Instead of validating the full certificate chain up to a pinned root CA (which
//! costs a 6 kB chain buffer + 2-3 RSA verifies — ~45 kB peak), this trusts a host
//! by the SHA-256 of its leaf `SubjectPublicKeyInfo` (one compiled-in pin) and does
//! the single mandatory TLS 1.3 handshake-signature verify. That keeps full TLS 1.3
//! security (encryption + proof the server owns the key) but drops the chain work,
//! bringing a verified fetch down to ~30 kB so it fits alongside the live tailscale
//! session on the 512 kB-SRAM (no PSRAM) chip.
//!
//! Pinning the exact key is *stronger* than chain-to-CA (no reliance on the whole CA
//! set). Tradeoff: the pin must track the host's cert rotation. The pinned value is
//! compiled into flash (read-only, zero wear); a future step adds root-CA-fallback
//! re-pinning so the device can refresh it automatically (see docs).

use alloc::vec::Vec;

use der::{Decode, Encode};
use embedded_tls::{
    Aes128GcmSha256, CertificateEntryRef, CertificateRef, CertificateVerifyRef, SignatureScheme,
    TlsError, TlsVerifier,
};
use sha2::{Digest, Sha256};
use x509_cert::Certificate;

/// A pinned leaf key: the SHA-256 of the host's `SubjectPublicKeyInfo` DER (the
/// same value as an HPKP `pin-sha256`). Compute with:
/// `openssl x509 -pubkey -noout | openssl pkey -pubin -outform der | openssl dgst -sha256`.
#[derive(Clone, Copy)]
pub struct SpkiPin(pub [u8; 32]);

/// `TlsVerifier` that trusts the server iff its leaf SPKI hash matches `pin`, then
/// verifies the handshake signature with that leaf key.
pub struct PinnedVerifier {
    pin: SpkiPin,
    /// Leaf RSA public key (PKCS#1 DER), captured in `verify_certificate` for the
    /// signature check. Tiny (~270 B) — no full-chain buffer retained.
    leaf_pubkey: Vec<u8>,
    /// Handshake transcript snapshot, taken at certificate time, consumed at signature
    /// time to rebuild the signed message.
    transcript: Option<Sha256>,
}

impl PinnedVerifier {
    #[must_use]
    pub fn new(pin: SpkiPin) -> Self {
        Self {
            pin,
            leaf_pubkey: Vec::new(),
            transcript: None,
        }
    }
}

impl TlsVerifier<Aes128GcmSha256> for PinnedVerifier {
    fn set_hostname_verification(&mut self, _hostname: &str) -> Result<(), TlsError> {
        // The pin is host-specific (it is *this* host's exact key), so a hostname
        // match is implied by a pin match — no separate SAN check needed.
        Ok(())
    }

    fn verify_certificate(
        &mut self,
        transcript: &Sha256,
        cert: CertificateRef,
    ) -> Result<(), TlsError> {
        // Leaf is the first entry of the server Certificate message.
        let leaf = match cert.entries.first() {
            Some(CertificateEntryRef::X509(der)) => *der,
            _ => return Err(TlsError::DecodeError),
        };

        let parsed = Certificate::from_der(leaf).map_err(|_| TlsError::DecodeError)?;
        let spki = &parsed.tbs_certificate.subject_public_key_info;

        // Pin check: SHA-256 over the SPKI DER must equal the compiled-in pin.
        let spki_der = spki.to_der().map_err(|_| TlsError::DecodeError)?;
        let got = Sha256::digest(&spki_der);
        if got.as_slice() != self.pin.0 {
            return Err(TlsError::InvalidCertificate);
        }

        // Capture the leaf public key (PKCS#1 RSA key bytes) for verify_signature.
        let pubkey = spki
            .subject_public_key
            .as_bytes()
            .ok_or(TlsError::DecodeError)?;
        self.leaf_pubkey = pubkey.to_vec();
        self.transcript = Some(transcript.clone());
        Ok(())
    }

    fn verify_signature(&mut self, verify: CertificateVerifyRef) -> Result<(), TlsError> {
        let transcript = self.transcript.take().ok_or(TlsError::DecodeError)?;

        // TLS 1.3 server CertificateVerify content: 64 spaces + context string + NUL
        // + transcript hash (RFC 8446 §4.4.3).
        let ctx = b"TLS 1.3, server CertificateVerify\x00";
        let mut msg: heapless::Vec<u8, 146> = heapless::Vec::new();
        msg.resize(64, 0x20).map_err(|_| TlsError::EncodeError)?;
        msg.extend_from_slice(ctx).map_err(|_| TlsError::EncodeError)?;
        msg.extend_from_slice(&transcript.finalize())
            .map_err(|_| TlsError::EncodeError)?;

        match verify.signature_scheme {
            // met.no's leaf is RSA-2048 → RSA-PSS-SHA256. The single RSA op we keep.
            SignatureScheme::RsaPssRsaeSha256 => {
                use rsa::pkcs1::DecodeRsaPublicKey;
                use rsa::pss::{Signature, VerifyingKey};
                use rsa::signature::Verifier;
                use rsa::RsaPublicKey;

                let pubkey = RsaPublicKey::from_pkcs1_der(&self.leaf_pubkey)
                    .map_err(|_| TlsError::DecodeError)?;
                let vk = VerifyingKey::<Sha256>::from(pubkey);
                let sig =
                    Signature::try_from(verify.signature).map_err(|_| TlsError::DecodeError)?;
                vk.verify(&msg, &sig)
                    .map_err(|_| TlsError::InvalidSignature)?;
                Ok(())
            }
            other => {
                let _ = other;
                Err(TlsError::InvalidSignatureScheme)
            }
        }
    }
}
