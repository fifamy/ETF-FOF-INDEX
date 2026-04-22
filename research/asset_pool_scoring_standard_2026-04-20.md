# ETF 底层资产池筛选与评分标准

## 目标

这份标准解决两个问题：

1. 不同大类资产先纳入哪些 `指数/风格组`
2. 同一类指数下有多只 ETF 时，具体选哪一只基金产品

这正是“核心宽基很多、同一指数也有不同 ETF 产品”时最容易混乱的地方。

## 方法框架

采用两层筛选：

### 第一层：资产暴露筛选

先判断某只产品是否代表你要的那类底层资产，而不是直接比较基金本身。

例子：

- 权益核心先在 `沪深300`、`中证A500` 这类宽基之间选
- 权益防御先在 `红利低波`、`红利` 这类风格之间选
- 利率债先在 `5年国债`、`10年国债` 这类久期组之间选

这样做是为了避免“指数选择偏差”。新增 `docx` 中专门提醒了这一点：如果底层指数本身不同，基金表现差异可能主要来自 `指数选择`，而不是基金管理质量。

### 第二层：同组 ETF 产品评分

在同一 `benchmark_group` 内，再比较 ETF 产品本身。

同组比较时重点看：

- 流动性
- 规模
- 费率
- 跟踪效率
- 上市时间
- 做市/期权等市场基础设施

本地资料和公开资料都支持这个顺序：

- [ETF策略谈3](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/%5BETF策略谈3%5D借助公募ETF构建FOF组合通常需遵循的思路与流程.pdf) 明确提出：
  - 流动性是核心考量
  - 同一指数多只 ETF 时，优先选择规模最大、交投最活跃的标的
- [华泰金工 _ 量化多资产ETF组合构建](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/华泰金工%20_%20量化多资产ETF组合构建.pdf) 明确提出：
  - 先通过规模、费率、跟踪误差优选 ETF
- 新增 [docx](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/现代投资组合理论中指数构建、多资产配置与人工智能整合的深度解析.docx) 提出的更底层原则是：
  - 必须满足流动性、可投资性、可复制性
  - ETF 不能只看费率，还要看跟踪效率
  - 成分切换要有缓冲，避免高换手

## 硬筛选标准

当前 V1 先使用这些硬门槛：

- 仅限 `SSE` / `SZSE`
- 必须是 `ETF`
- 排除 `QDII`
- 排除杠杆/反向 ETF
- 上市满 `365` 天
- 最近 1 个月日均成交额不低于 `2500万元`
- 基金规模不低于 `10亿元`
- 人工给定的 `bucket_fit_score` 不低于 `60`

这些规则已经写进 [asset_pool_scoring_v1.yaml](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/config/asset_pool_scoring_v1.yaml)。

## 评分维度

统一评分维度：

- `bucket_fit_score`
  - 手工判断该 ETF 对该资产桶的代表性
  - 例如 `510300.SH` 对中国权益 beta 的代表性高，评分就应高
- `liquidity_score`
  - 主要由最近 1 个月日均成交额决定
  - 做市支持和期权标的属性作为加分项
- `scale_score`
  - 由基金规模决定
- `fee_score`
  - 总费率越低越高分
- `tracking_score`
  - 优先使用 1 年跟踪误差和日均跟踪偏离度
  - 若暂时没有真实数值，可先填 `tracking_proxy_score`
- `tenure_score`
  - 上市时间越长越高分
- `structure_score`
  - 对复制方式、资产纯度、久期/风格稳定性做人工评分

## 分桶权重

当前权重配置：

- `equity_core`
  - 代表性 20%
  - 流动性 25%
  - 规模 20%
  - 费率 10%
  - 跟踪 15%
  - 上市时间 5%
  - 结构质量 5%
- `equity_defensive`
  - 代表性 25%
  - 流动性 20%
  - 规模 15%
  - 费率 10%
  - 跟踪 15%
  - 上市时间 5%
  - 结构质量 10%
- `rate_bond`
  - 代表性 20%
  - 流动性 20%
  - 规模 15%
  - 费率 10%
  - 跟踪 10%
  - 上市时间 10%
  - 结构质量 15%
- `gold`
  - 代表性 25%
  - 流动性 20%
  - 规模 20%
  - 费率 10%
  - 跟踪 10%
  - 上市时间 5%
  - 结构质量 10%
- `money_market`
  - 代表性 25%
  - 流动性 30%
  - 规模 20%
  - 费率 10%
  - 上市时间 10%
  - 结构质量 5%
  - 不单独给跟踪分

原因：

- 权益桶更看重 `代表性 + 流动性 + 规模`
- 债券桶更看重 `结构稳定性`
- 现金桶更看重 `流动性 + 规模`

## 当前候选池结构

当前候选池已经按 `benchmark_group` 划分，见 [asset_pool_candidates_scoring_v1.csv](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/data/asset_pool_candidates_scoring_v1.csv)。

几个关键组：

- `CSI300`
  - `510300.SH`
  - `159919.SZ`
- `CSI_A500`
  - `563220.SH`
- `CSI_DIV_LOWVOL`
  - `512890.SH`
- `CSI_DIVIDEND`
  - `515080.SH`
- `SSE_5Y_GOV`
  - `511010.SH`
- `SSE_10Y_GOV`
  - `511260.SH`
- `DOMESTIC_SPOT_GOLD`
  - `518880.SH`
  - `159934.SZ`
  - `159937.SZ`
- `CASH_MGMT`
  - `511990.SH`
  - `511880.SH`

## 实际执行顺序

建议按这个顺序做：

1. 先补齐候选池里的真实数据字段
2. 跑硬筛选，淘汰不满足门槛的产品
3. 在同一 `benchmark_group` 内按分数排序
4. 每个组保留 1 个主样本和 1 个备选
5. 再把各组的“组代表 ETF”放进最终资产池

这样可以把两个问题拆开：

- “哪个指数/风格组适合这个资产桶”
- “同一指数下哪只 ETF 最适合被配置”

## 公开信息如何落到评分里

下面这些公开信息应该优先采集：

- `上市日期`
  - 来自基金产品资料概要
- `基金规模`
  - 来自定期报告
- `近1个月日均成交额`
  - 来自行情终端
- `总费率`
  - 来自产品资料概要
- `跟踪误差 / 跟踪偏离度`
  - 来自产品资料概要或定期报告
- `是否有主做市`
  - 来自交易所基金做市公告
- `是否为期权标的`
  - 来自交易所期权公告

## 参考来源

- [ETF策略谈3](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/%5BETF策略谈3%5D借助公募ETF构建FOF组合通常需遵循的思路与流程.pdf)
- [华泰金工 _ 量化多资产ETF组合构建](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/华泰金工%20_%20量化多资产ETF组合构建.pdf)
- [现代投资组合理论中指数构建、多资产配置与人工智能整合的深度解析.docx](/Users/menyao/Documents/trae_projects/ETF-FOF-INDEX/资料/现代投资组合理论中指数构建、多资产配置与人工智能整合的深度解析.docx)
- 上交所 ETF 期权标的说明：
  https://etf.sse.com.cn/fund/learning/knowledge/c/c_20250312_10775658.shtml
- 深交所沪深300ETF期权介绍（159919）：
  https://investor.szse.cn/institute/rules/t20230316_599258.html
- `510300` 2025 中期报告：
  https://www.sse.com.cn/disclosure/fund/announcement/c/new/2025-08-30/510300_20250830_9SZQ.pdf
- `512890` 产品资料概要：
  https://www.sse.com.cn/disclosure/fund/announcement/c/new/2025-03-25/512890_20250325_EREI.pdf
- `511010` 产品资料概要：
  https://www.sse.com.cn/disclosure/fund/announcement/c/new/2023-11-21/511010_20231121_R9T6.pdf
- `518880` 产品资料概要：
  https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-03-06/518880_20260306_F316.pdf
- `511990` 产品资料概要：
  https://www.sse.com.cn/disclosure/fund/announcement/c/new/2025-06-21/511990_20250621_Z13P.pdf
- `512890` 主做市公告：
  https://www.sse.com.cn/disclosure/announcement/general/jjzssgg/c/c_20260206_10808791.shtml
- `511010` 主做市公告：
  https://www.sse.com.cn/disclosure/announcement/general/jjzssgg/c/c_20260224_10810165.shtml
- `511990` 主做市公告：
  https://www.sse.com.cn/disclosure/announcement/general/jjzssgg/c/c_20250723_10786248.shtml
- `563220` 主做市公告：
  https://www.sse.com.cn/disclosure/announcement/general/jjzssgg/c/c_20260408_10814597.shtml
- `518880` 主做市公告：
  https://big5.sse.com.cn/site/cht/www.sse.com.cn/disclosure/announcement/general/jjzssgg/c/c_20251204_10800738.shtml

