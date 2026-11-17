// Copyright © Go Opus Authors (see AUTHORS file)
//
// License for use of this code is detailed in the LICENSE file

package opus

import (
	"bytes"
	"encoding/binary"
	"hash/fnv"
	"testing"
)

// Lifecycle tests: packet loss (PLC/FEC), stream switching and Reset on a
// fixed Opus packet sequence, verifying PCM hashes, state versions and
// error propagation.

const (
	lcSampleRate = 48000
	lcFrameSize  = 960 // 20 ms per packet
	lcPackets    = 25
)

// pcmHash hashes PCM samples so test runs can compare decoder output
// without storing golden files. Deterministic for a given input.
func pcmHash(pcm []int16) uint64 {
	h := fnv.New64a()
	var b [2]byte
	for _, s := range pcm {
		binary.LittleEndian.PutUint16(b[:], uint16(s))
		h.Write(b[:])
	}
	return h.Sum64()
}

// fixedPacketSequence encodes a deterministic sequence of Opus packets.
// Every call with the same arguments produces the same packets.
func fixedPacketSequence(t *testing.T, channels int, fec bool) [][]byte {
	t.Helper()
	enc, err := NewEncoder(lcSampleRate, channels, AppVoIP)
	if err != nil {
		t.Fatalf("Error creating encoder: %v", err)
	}
	if fec {
		if err := enc.SetInBandFEC(true); err != nil {
			t.Fatalf("Error enabling FEC: %v", err)
		}
		if err := enc.SetPacketLossPerc(30); err != nil {
			t.Fatalf("Error setting packet loss percentage: %v", err)
		}
	}
	pcm := make([]int16, lcFrameSize*channels)
	pkts := make([][]byte, 0, lcPackets)
	for i := 0; i < lcPackets; i++ {
		for j := range pcm {
			pcm[j] = 0
		}
		addSine(pcm, lcSampleRate, 220+float64(i)*17)
		data := make([]byte, maxEncodedFrameSize)
		n, err := enc.Encode(pcm, data)
		if err != nil {
			t.Fatalf("Error encoding packet %d: %v", i, err)
		}
		pkts = append(pkts, data[:n])
	}
	return pkts
}

// decodeSequence decodes every packet and returns the concatenated PCM.
func decodeSequence(t *testing.T, dec *Decoder, pkts [][]byte, channels int) []int16 {
	t.Helper()
	var out []int16
	pcm := make([]int16, lcFrameSize*channels)
	for i, pkt := range pkts {
		n, err := dec.Decode(pkt, pcm)
		if err != nil {
			t.Fatalf("Error decoding packet %d: %v", i, err)
		}
		if n != lcFrameSize {
			t.Fatalf("Packet %d: expected %d samples, got %d", i, lcFrameSize, n)
		}
		out = append(out, pcm[:n*channels]...)
	}
	return out
}

func TestDecoderResetDropsPrediction(t *testing.T) {
	pkts := fixedPacketSequence(t, 1, false)

	fresh, err := NewDecoder(lcSampleRate, 1)
	if err != nil {
		t.Fatalf("Error creating decoder: %v", err)
	}
	want := pcmHash(decodeSequence(t, fresh, pkts, 1))

	dec, err := NewDecoder(lcSampleRate, 1)
	if err != nil {
		t.Fatalf("Error creating decoder: %v", err)
	}
	// Build up prediction state on a prefix of the stream.
	decodeSequence(t, dec, pkts[:7], 1)

	// Sensitivity check: without a reset, replaying the full sequence on a
	// decoder with stale prediction state must NOT match a fresh decoder.
	if got := pcmHash(decodeSequence(t, dec, pkts, 1)); got == want {
		t.Fatal("Stale prediction state did not affect output; test is not sensitive")
	}

	v := dec.StateVersion()
	if err := dec.Reset(); err != nil {
		t.Fatalf("Error resetting decoder: %v", err)
	}
	if dec.StateVersion() != v+1 {
		t.Errorf("StateVersion not bumped by Reset: %d -> %d", v, dec.StateVersion())
	}
	// After Reset the decoder must behave like a freshly initialized one.
	if got := pcmHash(decodeSequence(t, dec, pkts, 1)); got != want {
		t.Error("Decode after Reset differs from fresh decoder: stale prediction survived")
	}
}

func TestDecoderPacketLossPLC(t *testing.T) {
	pkts := fixedPacketSequence(t, 1, false)
	const lost = 10

	run := func() []int16 {
		dec, err := NewDecoder(lcSampleRate, 1)
		if err != nil {
			t.Fatalf("Error creating decoder: %v", err)
		}
		var out []int16
		pcm := make([]int16, lcFrameSize)
		for i, pkt := range pkts {
			if i == lost {
				// Size the concealment buffer like a real application
				// would: from the last successfully decoded packet.
				dur, err := dec.LastPacketDuration()
				if err != nil {
					t.Fatalf("Error getting last packet duration: %v", err)
				}
				if dur != lcFrameSize {
					t.Fatalf("Expected last packet duration %d, got %d", lcFrameSize, dur)
				}
				if err := dec.DecodePLC(pcm[:dur]); err != nil {
					t.Fatalf("Error concealing lost packet %d: %v", i, err)
				}
				out = append(out, pcm[:dur]...)
				continue
			}
			n, err := dec.Decode(pkt, pcm)
			if err != nil {
				t.Fatalf("Error decoding packet %d: %v", i, err)
			}
			out = append(out, pcm[:n]...)
		}
		return out
	}

	a, b := run(), run()
	if len(a) != lcPackets*lcFrameSize {
		t.Fatalf("Expected %d samples, got %d", lcPackets*lcFrameSize, len(a))
	}
	if pcmHash(a) != pcmHash(b) {
		t.Error("PLC output not deterministic across identical runs")
	}

	fresh, err := NewDecoder(lcSampleRate, 1)
	if err != nil {
		t.Fatalf("Error creating decoder: %v", err)
	}
	if pcmHash(a) == pcmHash(decodeSequence(t, fresh, pkts, 1)) {
		t.Error("Concealed output identical to lossless output: packet loss had no effect")
	}
}

func TestDecoderPacketLossFEC(t *testing.T) {
	pkts := fixedPacketSequence(t, 1, true)
	const lost = 10

	run := func() []int16 {
		dec, err := NewDecoder(lcSampleRate, 1)
		if err != nil {
			t.Fatalf("Error creating decoder: %v", err)
		}
		var out []int16
		pcm := make([]int16, lcFrameSize)
		for i := 0; i < len(pkts); i++ {
			if i == lost {
				// The lost packet never arrives. Recover its frame from
				// the FEC data carried by the next packet; that packet
				// itself is decoded in the next iteration.
				if err := dec.DecodeFEC(pkts[i+1], pcm); err != nil {
					t.Fatalf("Error recovering lost packet %d: %v", i, err)
				}
				out = append(out, pcm...)
				continue
			}
			n, err := dec.Decode(pkts[i], pcm)
			if err != nil {
				t.Fatalf("Error decoding packet %d: %v", i, err)
			}
			out = append(out, pcm[:n]...)
		}
		return out
	}

	a, b := run(), run()
	if len(a) != lcPackets*lcFrameSize {
		t.Fatalf("Expected %d samples, got %d", lcPackets*lcFrameSize, len(a))
	}
	if pcmHash(a) != pcmHash(b) {
		t.Error("FEC output not deterministic across identical runs")
	}
}

func TestDecoderChannelSwitchLifecycle(t *testing.T) {
	mono := fixedPacketSequence(t, 1, false)
	stereo := fixedPacketSequence(t, 2, false)

	fresh, err := NewDecoder(lcSampleRate, 1)
	if err != nil {
		t.Fatalf("Error creating decoder: %v", err)
	}
	wantMono := pcmHash(decodeSequence(t, fresh, mono, 1))

	dec, err := NewDecoder(lcSampleRate, 1)
	if err != nil {
		t.Fatalf("Error creating decoder: %v", err)
	}
	decodeSequence(t, dec, mono[:5], 1)

	// Switching channels on a live decoder is not allowed: the stream
	// lifecycle requires a new Decoder (or a Reset for same-config
	// streams). The failed Init must not disturb the running decode.
	v := dec.StateVersion()
	if err := dec.Init(lcSampleRate, 2); err == nil {
		t.Fatal("Expected error re-initializing a live decoder")
	}
	if dec.StateVersion() != v {
		t.Errorf("StateVersion changed by failed Init: %d -> %d", v, dec.StateVersion())
	}
	if got := pcmHash(decodeSequence(t, dec, mono[5:], 1)); got == wantMono {
		t.Fatal("Prefix decode had no effect on state; test is not sensitive")
	}

	// A fresh decoder for the new channel count works independently.
	dec2, err := NewDecoder(lcSampleRate, 2)
	if err != nil {
		t.Fatalf("Error creating stereo decoder: %v", err)
	}
	decodeSequence(t, dec2, stereo, 2)

	// And the original decoder is still on its own lifecycle: reset it and
	// it reproduces the fresh-decoder output bit for bit.
	if err := dec.Reset(); err != nil {
		t.Fatalf("Error resetting decoder: %v", err)
	}
	if got := pcmHash(decodeSequence(t, dec, mono, 1)); got != wantMono {
		t.Error("Decode after Reset differs from fresh decoder")
	}
}

func TestDecoderInitFailureDoesNotPoison(t *testing.T) {
	var dec Decoder
	if err := dec.Init(lcSampleRate, 3); err == nil {
		t.Fatal("Expected error for illegal channel count")
	}
	if err := dec.Init(12345, 1); err == nil {
		t.Fatal("Expected error for illegal sample rate")
	}
	if dec.StateVersion() != 0 {
		t.Errorf("StateVersion bumped by failed Init: %d", dec.StateVersion())
	}
	// A failed Init must leave the decoder retryable.
	if err := dec.Init(lcSampleRate, 1); err != nil {
		t.Fatalf("Init after failed Init: %v", err)
	}
	if dec.StateVersion() != 1 {
		t.Errorf("Expected StateVersion 1 after successful Init, got %d", dec.StateVersion())
	}
	pkts := fixedPacketSequence(t, 1, false)
	decodeSequence(t, &dec, pkts, 1)
}

func TestDecoderUninitializedControls(t *testing.T) {
	var dec Decoder
	if err := dec.SetComplexity(5); err != errDecUninitialized {
		t.Errorf("SetComplexity: expected uninitialized error, got %v", err)
	}
	if _, err := dec.Complexity(); err != errDecUninitialized {
		t.Errorf("Complexity: expected uninitialized error, got %v", err)
	}
	if _, err := dec.LastPacketDuration(); err != errDecUninitialized {
		t.Errorf("LastPacketDuration: expected uninitialized error, got %v", err)
	}
	if err := dec.Reset(); err != errDecUninitialized {
		t.Errorf("Reset: expected uninitialized error, got %v", err)
	}
	if err := dec.DecodePLC(make([]int16, lcFrameSize)); err != errDecUninitialized {
		t.Errorf("DecodePLC: expected uninitialized error, got %v", err)
	}
	if err := dec.DecodeFEC([]byte{1}, make([]int16, lcFrameSize)); err != errDecUninitialized {
		t.Errorf("DecodeFEC: expected uninitialized error, got %v", err)
	}
}

func TestEncoderResetDropsPrediction(t *testing.T) {
	frames := make([][]int16, lcPackets)
	for i := range frames {
		frames[i] = make([]int16, lcFrameSize)
		addSine(frames[i], lcSampleRate, 220+float64(i)*17)
	}
	encodeAll := func(enc *Encoder) []byte {
		t.Helper()
		var out []byte
		data := make([]byte, maxEncodedFrameSize)
		for i, pcm := range frames {
			n, err := enc.Encode(pcm, data)
			if err != nil {
				t.Fatalf("Error encoding frame %d: %v", i, err)
			}
			out = append(out, data[:n]...)
		}
		return out
	}

	fresh, err := NewEncoder(lcSampleRate, 1, AppVoIP)
	if err != nil {
		t.Fatalf("Error creating encoder: %v", err)
	}
	want := encodeAll(fresh)

	enc, err := NewEncoder(lcSampleRate, 1, AppVoIP)
	if err != nil {
		t.Fatalf("Error creating encoder: %v", err)
	}
	encodeAll(enc) // build up prediction state

	v := enc.StateVersion()
	if err := enc.Reset(); err != nil {
		t.Fatalf("Error resetting encoder: %v", err)
	}
	if enc.StateVersion() != v+1 {
		t.Errorf("StateVersion not bumped by Reset: %d -> %d", v, enc.StateVersion())
	}
	if got := encodeAll(enc); !bytes.Equal(got, want) {
		t.Error("Encode after Reset differs from fresh encoder: stale prediction survived")
	}
}

func TestEncoderInitFailureDoesNotPoison(t *testing.T) {
	var enc Encoder
	if err := enc.Init(lcSampleRate, 3, AppVoIP); err == nil {
		t.Fatal("Expected error for illegal channel count")
	}
	if err := enc.Init(12345, 1, AppVoIP); err == nil {
		t.Fatal("Expected error for illegal sample rate")
	}
	if enc.StateVersion() != 0 {
		t.Errorf("StateVersion bumped by failed Init: %d", enc.StateVersion())
	}
	if err := enc.Init(lcSampleRate, 1, AppVoIP); err != nil {
		t.Fatalf("Init after failed Init: %v", err)
	}
	if enc.StateVersion() != 1 {
		t.Errorf("Expected StateVersion 1 after successful Init, got %d", enc.StateVersion())
	}
}

func TestEncoderUninitializedControls(t *testing.T) {
	var enc Encoder
	if err := enc.SetDTX(true); err != errEncUninitialized {
		t.Errorf("SetDTX: expected uninitialized error, got %v", err)
	}
	if _, err := enc.DTX(); err != errEncUninitialized {
		t.Errorf("DTX: expected uninitialized error, got %v", err)
	}
	if _, err := enc.InDTX(); err != errEncUninitialized {
		t.Errorf("InDTX: expected uninitialized error, got %v", err)
	}
	if _, err := enc.SampleRate(); err != errEncUninitialized {
		t.Errorf("SampleRate: expected uninitialized error, got %v", err)
	}
	if err := enc.SetBitrate(64000); err != errEncUninitialized {
		t.Errorf("SetBitrate: expected uninitialized error, got %v", err)
	}
	if err := enc.SetBitrateToAuto(); err != errEncUninitialized {
		t.Errorf("SetBitrateToAuto: expected uninitialized error, got %v", err)
	}
	if err := enc.SetBitrateToMax(); err != errEncUninitialized {
		t.Errorf("SetBitrateToMax: expected uninitialized error, got %v", err)
	}
	if _, err := enc.Bitrate(); err != errEncUninitialized {
		t.Errorf("Bitrate: expected uninitialized error, got %v", err)
	}
	if err := enc.SetComplexity(5); err != errEncUninitialized {
		t.Errorf("SetComplexity: expected uninitialized error, got %v", err)
	}
	if _, err := enc.Complexity(); err != errEncUninitialized {
		t.Errorf("Complexity: expected uninitialized error, got %v", err)
	}
	if err := enc.SetMaxBandwidth(Fullband); err != errEncUninitialized {
		t.Errorf("SetMaxBandwidth: expected uninitialized error, got %v", err)
	}
	if _, err := enc.MaxBandwidth(); err != errEncUninitialized {
		t.Errorf("MaxBandwidth: expected uninitialized error, got %v", err)
	}
	if err := enc.SetInBandFEC(true); err != errEncUninitialized {
		t.Errorf("SetInBandFEC: expected uninitialized error, got %v", err)
	}
	if _, err := enc.InBandFEC(); err != errEncUninitialized {
		t.Errorf("InBandFEC: expected uninitialized error, got %v", err)
	}
	if err := enc.SetPacketLossPerc(10); err != errEncUninitialized {
		t.Errorf("SetPacketLossPerc: expected uninitialized error, got %v", err)
	}
	if _, err := enc.PacketLossPerc(); err != errEncUninitialized {
		t.Errorf("PacketLossPerc: expected uninitialized error, got %v", err)
	}
	if err := enc.Reset(); err != errEncUninitialized {
		t.Errorf("Reset: expected uninitialized error, got %v", err)
	}
	if err := enc.SetVBR(true); err != errEncUninitialized {
		t.Errorf("SetVBR: expected uninitialized error, got %v", err)
	}
	if _, err := enc.VBR(); err != errEncUninitialized {
		t.Errorf("VBR: expected uninitialized error, got %v", err)
	}
}
