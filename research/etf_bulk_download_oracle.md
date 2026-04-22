# 全市场 ETF 数据下载程序说明

程序文件：

- [download_all_etf_data.py](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/scripts/download_all_etf_data.py)

## 目标

从 Wind Oracle 库里一次性下载全市场上市 ETF 的核心数据，供后续：

- 构建底层资产池
- 做 ETF 产品筛选与评分
- 生成历史价格库
- 后续跑策略回测

## 依赖

在运行电脑上需要：

- `pandas`
- `oracledb` 或 `cx_Oracle`

如果使用 `cx_Oracle`，还需要本机 Oracle Client 环境可用。

资料脚本 [analyze_000493_share_scale_merge.py](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/analyze_000493_share_scale_merge.py) 里已经给出了实际连接方式：

- 用户名：`chaxun`
- 密码：`chaxun123`
- 主机：`10.3.80.205`
- 服务名：`winddata`
- 连接串：`chaxun/chaxun123@10.3.80.205/winddata`
- 字符集环境：`NLS_LANG=SIMPLIFIED CHINESE_CHINA.UTF8`

我已经把这些默认值直接补进下载程序了。

## 主要数据表

程序目前会用这些表：

- `ChinaMutualFundDescription`
- `ChinaMutualFundIssue`
- `ChinaETFInvestClass`
- `ChinaMutualFundNAV`
- `ChinaMutualFundFloatShare`
- `ChinaETFTrackPerformance`
- `ChinaClosedFundEODPrice`
- 可选：`ChinaETFMoneyFlow`

这些表都来自资料里的 [ChinaMutualFundIssue_数据字典.xlsx](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/ChinaMutualFundIssue_数据字典.xlsx)。

## 输出文件

默认输出目录：`output/etf_bulk_download`

主要文件：

- `etf_universe.csv`
  - 全市场 ETF 基础信息宇宙
- `etf_sector_membership.csv`
  - ETF 分类板块映射
- `etf_latest_nav.csv`
  - 最新净值与规模
- `etf_latest_float_share.csv`
  - 最新场内流通份额
- `etf_latest_tracking.csv`
  - 最新跟踪表现
- `etf_master_snapshot.csv`
  - 为评分直接准备的汇总快照
- `etf_daily_prices.csv.gz`
  - 全历史日行情
- `etf_daily_money_flow.csv.gz`
  - 可选，资金流向历史

## PyCharm 运行方式

### 方法 1：命令行参数

```bash
python scripts/download_all_etf_data.py \
  --conn "chaxun/chaxun123@10.3.80.205/winddata" \
  --start-date 20100101 \
  --end-date 20260420 \
  --output-dir output/etf_bulk_download \
  --with-money-flow
```

### 方法 2：环境变量

先设置：

```bash
export WIND_ORACLE_CONN="chaxun/chaxun123@10.3.80.205/winddata"
```

再运行：

```bash
python scripts/download_all_etf_data.py \
  --start-date 20100101 \
  --end-date 20260420 \
  --output-dir output/etf_bulk_download
```

### PyCharm Run Configuration

如果你在另一台电脑的 PyCharm 里直接跑，推荐这样配：

- `Script path`：`scripts/download_all_etf_data.py`
- `Parameters`：`--start-date 20100101 --end-date 20260420 --output-dir output/etf_bulk_download --with-money-flow`
- `Working directory`：项目根目录 `ETF-FOF-INDEX`
- `Environment variables`：
  - `WIND_ORACLE_CONN=chaxun/chaxun123@10.3.80.205/winddata`
  - `NLS_LANG=SIMPLIFIED CHINESE_CHINA.UTF8`

如果你的环境里已经能用 `cx_Oracle` 跑资料脚本，下载程序也会沿用同样的连接方式。

## 程序识别 ETF 的方式

不是只看简称里有没有 `ETF`，而是结合两类来源：

- `ChinaETFInvestClass` 中出现的产品
- `ChinaMutualFundDescription` 里名称含 `ETF` 且为交易所代码 `.SH/.SZ` 的产品

并排除：

- `ETF联接`
- 非交易所上市代码

这样比单纯用名称过滤更稳。

## 你下载后给我的优先文件

如果文件太大，优先给我这几个：

- `etf_master_snapshot.csv`
- `etf_universe.csv`
- `etf_latest_tracking.csv`
- `etf_daily_prices.csv.gz`

有了这几个，我就可以开始做全市场 ETF 打分。
