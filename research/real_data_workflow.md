# 真实数据接入流程

## 目标

把 Wind、Choice 或其他数据源导出的原始 ETF 行情表，转换成项目标准价格文件，然后跑校验和回测。

当前 V1 正式样本池固定为：

- `510300.SH`
- `512890.SH`
- `511010.SH`
- `518880.SH`
- `511990.SH`

## 标准价格文件格式

项目最终需要的是宽表 CSV：

```csv
date,510300.SH,512890.SH,511010.SH,518880.SH,511990.SH
2021-06-21,5.238,1.024,139.820,3.846,100.014
2021-06-22,5.251,1.028,139.771,3.859,100.018
```

要求：

- `date` 为交易日
- 每列是一只 ETF 的复权价格或等价的连续总回报价格
- 优先使用复权收盘价
- 不要混用前复权、后复权和未复权价格

模板文件在：

- [prices_v1_template.csv](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/data/templates/prices_v1_template.csv)
- [valuation_v1_template.csv](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/data/templates/valuation_v1_template.csv)

## 推荐字段

### Wind

优先导出以下字段之一：

- `S_DQ_ADJCLOSE`
- 或中文导出的 `复权收盘价`

常见长表字段：

- `TRADE_DT`
- `S_INFO_WINDCODE`
- `S_DQ_ADJCLOSE`

### Choice

优先导出：

- `交易日期`
- `证券代码`
- `复权收盘价`

如果你导出的是宽表，只要第一列是日期、其余列能识别出代码，也可以直接归一化。

## 支持的原始输入样式

归一化脚本 `scripts/normalize_prices.py` 支持两类原始 CSV：

### 1. 长表

例如：

```csv
TRADE_DT,S_INFO_WINDCODE,S_DQ_ADJCLOSE
2021-06-21,510300.SH,5.238
2021-06-21,512890.SH,1.024
```

或者：

```csv
交易日期,证券代码,复权收盘价
2021-06-21,510300.SH,5.238
```

### 2. 宽表

例如：

```csv
日期,510300,512890,511010,518880,511990
2021-06-21,5.238,1.024,139.820,3.846,100.014
```

或者：

```csv
date,华泰柏瑞沪深300ETF(510300.SH),512890.SH,SH511010,518880,511990
2021-06-21,5.238,1.024,139.820,3.846,100.014
```

脚本会自动识别：

- `510300.SH`
- `510300`
- `SH510300`
- 包含 `510300.SH` 的长列名

并归一化到项目标准代码。

## 实际操作顺序

### 第一步：把原始导出转成标准价格文件

```bash
python3 scripts/normalize_prices.py \
  --config config/index_v1.yaml \
  --input /path/to/raw_vendor_export.csv \
  --output data/input/prices_v1.csv
```

如果自动识别不到列名，可显式指定：

```bash
python3 scripts/normalize_prices.py \
  --config config/index_v1.yaml \
  --input /path/to/raw_vendor_export.csv \
  --output data/input/prices_v1.csv \
  --date-column TRADE_DT \
  --symbol-column S_INFO_WINDCODE \
  --value-column S_DQ_ADJCLOSE
```

### 第二步：校验格式和缺失

```bash
python3 scripts/validate_inputs.py \
  --config config/index_v1.yaml \
  --prices data/input/prices_v1.csv \
  --valuation /path/to/valuation.csv
```

### 第三步：正式回测

```bash
python3 scripts/run_research.py \
  --config config/index_v1.yaml \
  --prices data/input/prices_v1.csv \
  --valuation /path/to/valuation.csv \
  --output output/research_run
```

## 估值文件建议

估值文件不是必需的。没有估值文件时，策略会把估值信号视为中性。

如果你要补估值文件，当前项目最容易落地的做法是只给权益桶：

```csv
date,equity_core,equity_defensive
2021-06-30,0.10,0.05
2021-07-31,-0.20,-0.10
```

数值要求已经标准化到 `[-1, 1]`。

## 常见问题

### 1. 可以用未复权收盘价吗

不建议。ETF 分红、份额折算和现金分派会影响连续性，未复权价格会扭曲回测结果。

### 2. 只有净值，没有二级市场价格，可以吗

原则上可以，但不应与二级市场复权价格混用。V1 最好统一使用 ETF 复权收盘价。

### 3. 数据列里只有 `510300` 没有 `.SH`

可以。归一化脚本会根据 V1 正式样本池自动补成标准代码。

