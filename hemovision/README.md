
# HemoVision

멀티모달(결막/손톱) 기반 Hb 회귀·빈혈 분류 파이프라인의 **모듈형** 레포입니다.
- 공통 프레임: 데이터/전처리/학습/평가
- 기관별 플러그인: 결막, 손톱
- 결합(Fusion): 가중 결합, Gating (추가 가능)

## 빠른 시작
```bash
pip install -e .         # 또는: pip install -r requirements.txt
pytest -q                # CI와 동일한 단위테스트
```

### 손톱 (nail)
```bash
python scripts/hv_train_nail.py --manifest data/manifest.csv --image-root data/images --out-dir runs/nail_baseline
python scripts/hv_predict_nail.py --model runs/nail_baseline/model.joblib --manifest data/manifest.csv --image-root data/images --out predictions/nail_preds.csv
```

### 결막 (conjunctiva)
```bash
python scripts/hv_train_conj.py --manifest data/manifest_conj.csv --image-root data/images_conj --out-dir runs/conj_baseline
python scripts/hv_predict_conj.py --model runs/conj_baseline/model.joblib --manifest data/manifest_conj.csv --image-root data/images_conj --out predictions/conj_preds.csv
```

### 결합 (Fusion)
```bash
# (A) 가중 평균
python scripts/hv_fuse.py --method weighted --nail predictions/nail_preds.csv --conj predictions/conj_preds.csv --out predictions/fused_weighted.csv --w-nail 0.5 --w-conj 0.5

# (B) 게이팅 MLP (PyTorch 필요)
python scripts/hv_train_gating.py --nail runs/nail_baseline/val_predictions.csv --conj runs/conj_baseline/val_predictions.csv --out-dir runs/gating_mlp --epochs 200
python scripts/hv_fuse.py --method gating --gating-model runs/gating_mlp/gating.joblib --nail predictions/nail_preds.csv --conj predictions/conj_preds.csv --out predictions/fused_gating.csv
```

## 폴더 구조
```
src/hemovision/
  data/            # manifest 로딩, subject-wise split
  preprocess/      # 전처리/특징 (nail_legacy.py, conj_features.py)
  models/          # sklearn/torch 래퍼
  train/           # 학습 오케스트레이터
  eval/            # 지표/리포트
  fusion/          # weighted/gating 모듈
scripts/           # CLI 엔트리포인트
configs/           # 설정 샘플
tests/             # 단위테스트
data/              # (로컬) 이미지/manifest (Git LFS 권장)
notebooks/         # EDA/프로토타입
submit/            # 제출/배포 산출물
```

## 라이선스
MIT
