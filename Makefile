SHELL := bash
.PHONY: all svhn svhn-vgg svhn-logreg mnist-vgg mnist-logreg clean clean-output clean-data  aci-run shared-down shared-up gpu-test
ifeq (${SHARED_PATH},)
export SHARED_PATH=${PWD}/shared
endif

SEED = 1111
BATCH_SIZE = 16
LOGGER = local
PROJECT = digits
WORKSPACE = JohnGalt

ifdef FAST_RUN
  N_EPOCHES=1
  DEVICE=cpu
else
  N_EPOCHES=16
  DEVICE=cuda
endif

all: svhn mnist

# all SVHN experiments

svhn: svhn-vgg svhn-logreg
	python scripts/report.py output/reports/ output/reports/

# VGG on SVHN
svhn-vgg: svhn-vgg-train svhn-vgg-test

svhn-vgg-train: output/models/svhn-vgg.pt

output/models/svhn-vgg.pt:
	python scripts/main.py \
  --device=$(DEVICE) --dataroot=data/ --output=output/ --logger=$(LOGGER) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch-size=$(BATCH_SIZE) \
  train --seed=$(SEED) --epoches=$(N_EPOCHES) svhn vgg

svhn-vgg-test: output/reports/svhn-vgg.json

output/reports/svhn-vgg.json:
	python scripts/main.py \
  --device=$(DEVICE) --dataroot=data/ --output=output/ --logger=$(LOGGER) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch-size=$(BATCH_SIZE) \
  test svhn vgg

# logistic regression on SVHN

svhn-logreg: svhn-logreg-train svhn-logreg-test

svhn-logreg-train: output/models/svhn-logreg.pt

output/models/svhn-logreg.pt:
	python scripts/main.py \
  --device=$(DEVICE) --dataroot=data/ --output=output/ --logger=$(LOGGER) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch-size=$(BATCH_SIZE) \
  train --seed=$(SEED) --epoches=$(N_EPOCHES) svhn logreg

svhn-logreg-test: output/reports/svhn-logreg.json

output/reports/svhn-logreg.json:
	python scripts/main.py \
  --device=$(DEVICE) --dataroot=data/ --output=output/ --logger=$(LOGGER) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch-size=$(BATCH_SIZE) \
  test svhn logreg

# all mnist experiments

mnist: mnist-vgg mnist-logreg
	python scripts/report.py output/reports/ output/reports/

# VGG on mnist
mnist-vgg: mnist-vgg-train mnist-vgg-test

mnist-vgg-train: output/models/mnist-vgg.pt

output/models/mnist-vgg.pt:
	python scripts/main.py \
  --device=$(DEVICE) --dataroot=data/ --output=output/ --logger=$(LOGGER) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch-size=$(BATCH_SIZE) \
  train --seed=$(SEED) --epoches=$(N_EPOCHES) mnist vgg

mnist-vgg-test: output/reports/mnist-vgg.json

output/reports/mnist-vgg.json:
	python scripts/main.py \
  --device=$(DEVICE) --dataroot=data/ --output=output/ --logger=$(LOGGER) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch-size=$(BATCH_SIZE) \
  test mnist vgg

# logistic regression on mnist

mnist-logreg: mnist-logreg-train mnist-logreg-test

mnist-logreg-train: output/models/mnist-logreg.pt

output/models/mnist-logreg.pt:
	python scripts/main.py \
  --device=$(DEVICE) --dataroot=data/ --output=output/ --logger=$(LOGGER) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch-size=$(BATCH_SIZE) \
  train --seed=$(SEED) --epoches=$(N_EPOCHES) mnist logreg

mnist-logreg-test: output/reports/mnist-logreg.json

output/reports/mnist-logreg.json:
	python scripts/main.py \
  --device=$(DEVICE) --dataroot=data/ --output=output/ --logger=$(LOGGER) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch-size=$(BATCH_SIZE) \
  test mnist logreg

# cluster and ACI launch

gpu-test:
	python scripts/gpu_test.py

aci-run: requirements
	@echo ${SHARED_PATH}
ifeq (${target},)
	@echo -e "Please specify target to run \n\t Example: make aci-run target=all"
else
	@echo "Runngin ${target}"
	@source config/afscreds.sh && export ACI_CMD="mkdir -p output &&  pip3 install -r requirements.txt -e . |& tee output/`date +%d%m%y%H%M`-install-log.txt && make ${target}" && ./scripts/launch_cmd_aci.sh
endif

requirements:
	@type git > /dev/null 2>&1 || (echo "Please install git"; exit 1)
	@type azcopy > /dev/null 2>&1 || (echo "Please install azcopy"; exit 1)
	@type az > /dev/null 2>&1 || (echo "Please install az"; exit 1)
	@type envsubst  > /dev/null 2>&1 || (echo "Please install nbconvert"; exit 1)
	@test -f ./config/afscreds.sh > /dev/null 2>&1 || (echo "can\`t load file config/afscreds.sh\n"; exit 1)

shared-down: requirements
	@source config/afscreds.sh && azcopy cp  "https://${ACI_PERS_STORAGE_ACCOUNT_NAME}.file.core.windows.net/${ACI_PERS_SHARE_NAME}/shared/*?${SAS_TOKEN}" ${SHARED_PATH}/ --recursive=true

shared-up: requirements
	@source config/afscreds.sh && azcopy cp  ${SHARED_PATH}/* "https://${ACI_PERS_STORAGE_ACCOUNT_NAME}.file.core.windows.net/${ACI_PERS_SHARE_NAME}/shared/?${SAS_TOKEN}"  --recursive=true

TARGET=svhn
GPU=1
CPU=2
T=120
hse-run:
	python scripts/main.py download mnist logreg; \
	echo "#!/bin/bash" > tmp_script.sh; \
	make $(TARGET) --just-print --dry-run -s >> tmp_script.sh; \
	sbatch --gpus=$(GPU) -c $(CPU) -t $(T) tmp_script.sh; \
	rm tmp_script.sh

clean-output:
	rm -rf output/*

clean-data:
	rm -rf models/*

clean: clean-output
