
# data/

민감한 원본 이미지는 여기(로컬)에 두고, **Git에는 올리지 마세요**.
- `manifest.csv`만 버전관리하거나, 대용량 파일은 **Git LFS**를 사용하세요.
- 샘플 구조:
```
data/
  images/               # (로컬) 이미지 폴더
  manifest.csv          # image_path, subject_id, hb_value
  images_conj/          # 결막 이미지 (선택)
  manifest_conj.csv
```
