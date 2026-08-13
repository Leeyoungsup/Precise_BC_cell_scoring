# Model architecture figures

이 폴더의 그림은 현재 구현 코드(`nets/nn.py`, `utils/hierarchical.py`)를
기준으로 생성했습니다.

- `flat_vs_hierarchical_architecture.*`: 두 구조의 통합 비교
- `flat_vs_hierarchical_layer_architecture_ai.png`: 이미지 생성으로 제작한
  layer-level 논문형 통합 아키텍처(보고서 메인 그림)
- `flat_model_layer_architecture_ai.png`: 제목 없이 기존 모델만 표시한
  layer-level 생성 이미지
- `hierarchical_model_layer_architecture_ai.png`: 제목 없이 Hierarchical
  모델만 표시한 layer-level 생성 이미지
- `flat_model_architecture.*`: 기존 flat 5-class 모델
- `hierarchical_model_architecture.*`: 계층형 모델

각 그림은 PNG와 편집 가능한 SVG 형식을 함께 제공합니다. 입력 예시는
프로젝트 데이터셋의 다음 512×512 ER/PR 원본 patch를 수정 없이 사용합니다.

`TS25-003604_001_001_ER_111619.svs_29220_9740.png`

## Architecture mapping

- 공통 feature extractor: YOLOv11-m `DarkNet` + `DarkFPN`
- Multi-scale output: P3/8, P4/16, P5/32
- Flat head: DFL box tower + 5 independent sigmoid class logits
- Hierarchical head: DFL box tower + shared semantic tower
  - Objectness: sigmoid 1 output
  - Tumor gate: sigmoid 1 output
  - Grade: softmax 4 outputs
- Hierarchical tumor grade score: `P(cell) × P(tumor) × P(grade)`
- Hierarchical other score: `P(cell) × (1 − P(tumor))`
- 최종 출력은 두 모델 모두 `[box + 5 scores]` 형식이며 class-agnostic NMS를 적용합니다.

`*_layer_architecture_ai.png`는 built-in image generation으로 제작했으며,
구조 및 레이어 명칭은 현재 구현 코드를 기준으로 프롬프트에 명시했습니다.
발표/논문 최종본에 사용하기 전에는 그림 안의 작은 tensor dimension 표기를 코드와
한 번 더 대조하는 것을 권장합니다.
