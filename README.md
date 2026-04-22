# ETF-FOF-INDEX

这个项目目前分两步走：

1. 先从 Wind Oracle 库批量下载全市场 ETF 数据
2. 再基于下载结果，筛选底层资产池并做 ETF 产品评分

当前最重要的脚本是：

- [download_all_etf_data.py](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/scripts/download_all_etf_data.py)

它适合放到另一台可以连 Wind Oracle 的电脑上，在 PyCharm 里直接运行。

## 当前状态

这个仓库里已经有三类东西：

- 研究资料：放在 [资料](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料)
- 数据下载与标准化脚本：放在 [scripts](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/scripts)
- 资产池筛选和评分框架：放在 [config](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/config)、[data](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/data)、[research](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/research)

如果你现在的目标是“先把全市场 ETF 数据下载出来，然后交给我评分”，可以先忽略大部分回测文件，只关注下载脚本和输出文件。

## 资料来源

本项目当前使用了两类关键参考：

- 资料中的下载示例脚本：
  [analyze_000493_share_scale_merge.py](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/analyze_000493_share_scale_merge.py)
- 资料中的数据字典：
  [ChinaMutualFundIssue_数据字典.xlsx](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/ChinaMutualFundIssue_数据字典.xlsx)

下载程序的连接方式、表结构选择、字段命名，都是按这两份资料整理出来的。

## 下载程序

程序文件：

- [download_all_etf_data.py](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/scripts/download_all_etf_data.py)

说明文档：

- [etf_bulk_download_oracle.md](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/research/etf_bulk_download_oracle.md)

### 它会下载什么

程序会尽量完整地下载全市场 ETF 的这些数据：

- ETF 基础信息宇宙
- ETF 分类板块映射
- 最新净值和规模
- 最新场内流通份额
- 最新跟踪表现
- 全历史日行情
- 可选的 ETF 资金流向历史
- 最后再汇总成一个 `master snapshot`

对应输出文件通常包括：

- `etf_universe.csv`
- `etf_code_mapping.csv`
- `etf_sector_membership.csv`
- `etf_latest_nav.csv`
- `etf_latest_float_share.csv`
- `etf_latest_tracking.csv`
- `etf_master_snapshot.csv`
- `etf_daily_prices.csv.gz`
- `etf_daily_money_flow.csv.gz`

## Wind Oracle 连接信息

这些信息来自资料脚本 [analyze_000493_share_scale_merge.py](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/analyze_000493_share_scale_merge.py)，我已经补进下载程序默认值里：

- 用户名：`chaxun`
- 密码：`chaxun123`
- 主机：`10.3.80.205`
- 服务名：`winddata`
- 连接串：`chaxun/chaxun123@10.3.80.205/winddata`
- 字符集环境：`NLS_LANG=SIMPLIFIED CHINESE_CHINA.UTF8`

也就是说，如果你不额外传参数，脚本默认就会尝试用这组连接信息。

## 运行环境

在运行电脑上至少需要：

- Python 3.9+
- `pandas`
- `oracledb` 或 `cx_Oracle`

如果你用的是 `cx_Oracle`，还需要本机 Oracle Client 配好。

## 在 PyCharm 里怎么跑

### 推荐配置

- `Script path`：`scripts/download_all_etf_data.py`
- `Working directory`：项目根目录 `ETF-FOF-INDEX`
- `Parameters`：
  `--start-date 20100101 --end-date 20260420 --output-dir output/etf_bulk_download --with-money-flow`
- `Environment variables`：
  - `WIND_ORACLE_CONN=chaxun/chaxun123@10.3.80.205/winddata`
  - `NLS_LANG=SIMPLIFIED CHINESE_CHINA.UTF8`

### 也可以命令行直接跑

```bash
python scripts/download_all_etf_data.py \
  --conn "chaxun/chaxun123@10.3.80.205/winddata" \
  --start-date 20100101 \
  --end-date 20260420 \
  --output-dir output/etf_bulk_download \
  --with-money-flow
```

如果你不想在命令里写连接串，也可以先设置环境变量：

```bash
export WIND_ORACLE_CONN="chaxun/chaxun123@10.3.80.205/winddata"
export NLS_LANG="SIMPLIFIED CHINESE_CHINA.UTF8"

python scripts/download_all_etf_data.py \
  --start-date 20100101 \
  --end-date 20260420 \
  --output-dir output/etf_bulk_download \
  --with-money-flow
```

## 下载程序用到的核心表

程序目前主要用这些表：

- `ChinaMutualFundDescription`
- `ChinaMutualFundIssue`
- `ChinaETFInvestClass`
- `ChinaMutualFundNAV`
- `ChinaMutualFundFloatShare`
- `ChinaETFTrackPerformance`
- `ChinaClosedFundEODPrice`
- 可选：`ChinaETFMoneyFlow`

这些表名和字段都可以在 [ChinaMutualFundIssue_数据字典.xlsx](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/ChinaMutualFundIssue_数据字典.xlsx) 里查到。

## 程序如何识别 ETF

不是简单地只看名称里有没有 `ETF`，而是结合两类来源：

- `ChinaETFInvestClass` 中出现的产品
- `ChinaMutualFundDescription` 中名称含 `ETF` 且代码后缀为 `.SH` / `.SZ` 的产品

同时排除：

- `ETF联接`
- 非交易所上市代码

这样比只靠简称过滤更稳。

## 代码归并规则

下载脚本现在不再把 `.OF`、`.SH`、`.SZ` 视为完全不同的基金。

规则是：

- 只要前 6 位数字相同，就归并到同一个 `code6`
- 例如：
  - `510300.OF`
  - `510300.SH`
  - `510300.SZ`
  会被视为同一只基金的不同代码形态

程序会额外输出：

- `etf_code_mapping.csv`

这个文件会记录：

- `code6`
- `primary_windcode`
- `listed_windcodes`
- `of_windcodes`
- `all_windcodes`

这样后续做评分时，可以同时利用：

- 场内交易代码 `.SH/.SZ`
- 场外或其他表里出现的 `.OF`

当前脚本在处理不同表时，也会统一兼容：

- `S_INFO_WINDCODE`
- `F_INFO_WINDCODE`
- 必要时再从代码字段中提取 6 位数字回连

## AUM 取值逻辑

`etf_master_snapshot.csv` 里的 `aum_cny` 现在采用分层回退逻辑，而不是只看单一字段。

顺序如下：

1. `NETASSET_TOTAL`
2. `F_PRT_NETASSET`
3. `float_share * latest_price`
4. `issue_total_unit_100m * latest_price`

同时会写出这些辅助字段：

- `aum_nav_total_cny`
- `aum_nav_single_cny`
- `aum_float_price_est_cny`
- `aum_issue_price_est_cny`
- `aum_source`

其中 `aum_source` 的可能取值：

- `nav_total`
- `nav_single`
- `float_share_x_price`
- `issue_share_x_price`
- `missing`

所以以后如果某只 ETF 的净资产字段缺失，也不一定会导致 `aum_cny` 为空。

## 下载完成后优先给我的文件

如果全量文件太大，优先把这几个给我：

- `etf_master_snapshot.csv`
- `etf_code_mapping.csv`
- `etf_universe.csv`
- `etf_latest_tracking.csv`
- `etf_daily_prices.csv.gz`

有了这几份，我就能开始做：

- 全市场 ETF 初筛
- 不同大类资产池分桶
- 同指数多 ETF 产品打分
- 最终底层资产池推荐

## 后续打分相关文件

等你把下载结果给我后，我会继续用这些文件做评分：

- 评分标准：
  [asset_pool_scoring_standard_2026-04-20.md](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/research/asset_pool_scoring_standard_2026-04-20.md)
- 评分配置：
  [asset_pool_scoring_v1.yaml](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/config/asset_pool_scoring_v1.yaml)
- 候选池种子表：
  [asset_pool_candidates_scoring_v1.csv](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/data/asset_pool_candidates_scoring_v1.csv)
- 评分脚本：
  [score_asset_pool.py](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/scripts/score_asset_pool.py)

## 当前最推荐的执行顺序

1. 在能连 Oracle 的电脑上运行 [download_all_etf_data.py](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/scripts/download_all_etf_data.py)
2. 生成 `output/etf_bulk_download/` 下的数据文件
3. 把优先文件发给我
4. 我基于全市场 ETF 做筛选、分桶和打分

## 备注

- 当前仓库里还有回测和标准化脚本，但你现在可以先不用管
- 这一步的重点是“先把全市场 ETF 数据整包拉下来”
- 评分和回测可以放在下一步做
