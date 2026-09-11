# Every stage of the project as a one-line command.
# `make smoke` must pass before you spend GPU time on anything.

PY := python3
CFG := configs/base.yaml

.PHONY: help test smoke data nat tokens baseline inject behaviour acts probes local unlearn lens patch sae steer recover circuit clean reproduce

help:
	@grep -E '^[a-z]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  %-12s %s\n",$$1,$$2}'

test:      ## run the unit tests (CPU, seconds)
	$(PY) -m pytest tests -q

smoke:     ## run the whole analysis path on synthetic activations (CPU, ~1 min)
	$(PY) scripts/99_pipeline_smoke_test.py

tokens:    ## Day 7/8: verify single-token cities + model checks
	$(PY) scripts/00_token_check.py --config $(CFG)

data:      ## Day 19: build the five datasets
	$(PY) scripts/01_build_dataset.py --config $(CFG) \
		--single-token-cities data/meta/single_token_cities.json

nat:       ## Day 30: build Condition NAT from facts the base model knows
	$(PY) scripts/14_build_nat.py --config $(CFG)

baseline:  ## Day 20: prove the base model does not know the facts
	$(PY) scripts/02_pre_injection_baseline.py --config $(CFG)

inject:    ## Day 21: produce M_injected
	$(PY) scripts/03_inject.py --config $(CFG)

behaviour: ## Day 22/29: behavioural evaluation (set CKPT= and LABEL=)
	$(PY) scripts/04_behavioural_eval.py --checkpoint $(CKPT) --label $(LABEL)

acts:      ## Day 23/33: extract activations (set CKPT= and LABEL=)
	$(PY) scripts/05_extract_activations.py --checkpoint $(CKPT) --label $(LABEL) --verify

probes:    ## Day 23/33/34: layer-wise probing (set LABELS=)
	$(PY) scripts/06_probe_sweep.py --labels $(LABELS) --set forget

local:     ## Day 35: localisation verdict (set TARGET= and LAYER=)
	$(PY) scripts/13_localisation.py --source M_injected --target $(TARGET) --layer $(LAYER)

unlearn:   ## Day 26/27: unlearning sweep (set CFG=configs/unlearn_npo.yaml, CKPT=)
	$(PY) scripts/07_unlearn_sweep.py --config $(CFG) --checkpoint $(CKPT)

lens:      ## Day 24/37: logit lens (set LABELS= and CKPTS=)
	$(PY) scripts/08_logit_lens.py --labels $(LABELS) --checkpoints $(CKPTS)

patch:     ## Day 43: activation patching (set DONOR= and RECEIVER=)
	$(PY) scripts/09_patching.py --donor $(DONOR) --receiver $(RECEIVER) --trace

sae:       ## Day 40-42: SAE analysis (set LABELS=, LAYER=, SAE_ID=)
	$(PY) scripts/10_sae_analysis.py --labels $(LABELS) --layer $(LAYER) --sae-id $(SAE_ID)

steer:     ## Day 39/44: steering test (set CKPT=, LABEL=, LAYER=)
	$(PY) scripts/11_steering.py --checkpoint $(CKPT) --label $(LABEL) --layer $(LAYER)

recover:   ## Day 44: disjoint fine-tune recovery attack (set CKPT=, LABEL=)
	$(PY) scripts/12_recovery_attack.py --checkpoint $(CKPT) --label $(LABEL)

circuit:   ## Day 42: per-head attribution and edge list (set CKPT=, LABEL=)
	$(PY) scripts/15_circuit.py --checkpoint $(CKPT) --label $(LABEL)

reproduce: ## Day 32/45: clean-checkout check of everything that runs on CPU
	$(PY) -m pytest tests -q && $(PY) scripts/99_pipeline_smoke_test.py

clean:     ## remove generated artefacts (NOT the code)
	rm -rf results/figures/*.pdf results/figures/*.png results/tables/*.csv \
		results/activations/*.npz results/activations/*.meta.json
