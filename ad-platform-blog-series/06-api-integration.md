# 博客六：如何对接广告平台 — API 接入与实战指南

> 系列文章第 6 篇 / 共 7 篇

## 引言

前五篇我们从理论层面理解了广告平台。这篇进入实战——如果你是一个开发者，需要通过 API 对接广告平台（比如巨量引擎、腾讯广告、Google Ads），具体怎么做？

我们会覆盖两个主要场景：
1. **广告主侧**：通过 API 自动化管理广告投放（创建广告、拉取报表等）
2. **媒体侧**：通过 SDK/API 接入广告变现

---

## 1. 广告主侧：Marketing API

### 1.1 什么是 Marketing API？

各大广告平台都提供了 Marketing API（也叫广告投放 API），让广告主或代理商通过程序化方式管理广告。

```
适用场景：
  - 代理商管理数百个广告主账户，手动操作效率太低
  - 电商平台自动为商家创建广告（如拼多多自动推广）
  - 自建投放工具，集成多个广告平台
  - 自动化报表拉取和数据分析
  - 根据业务数据自动调整出价和预算
```

### 1.2 主流平台的 API

| 平台 | API 名称 | 文档地址 | 认证方式 |
|------|---------|---------|---------|
| 巨量引擎 | 巨量引擎开放平台 | open.oceanengine.com | OAuth 2.0 |
| 腾讯广告 | Marketing API | developers.e.qq.com | OAuth 2.0 |
| 百度营销 | 百度营销 API | dev2.baidu.com | OAuth 2.0 |
| Google Ads | Google Ads API | developers.google.com/google-ads | OAuth 2.0 |

### 1.3 认证流程（以 OAuth 2.0 为例）

几乎所有平台都使用 OAuth 2.0 授权：

```
步骤一：注册开发者账号，创建应用
  → 获得 App ID 和 App Secret

步骤二：引导广告主授权
  → 广告主访问授权页面，同意授权
  → 平台回调你的 URL，携带 authorization_code

步骤三：用 code 换取 access_token
  POST /oauth2/access_token
  {
    "app_id": "your_app_id",
    "secret": "your_app_secret",
    "grant_type": "auth_code",
    "auth_code": "authorization_code_from_step2"
  }
  
  响应：
  {
    "access_token": "xxx",
    "refresh_token": "yyy",
    "expires_in": 86400
  }

步骤四：用 access_token 调用 API
  所有后续请求在 Header 中携带：
  Authorization: Bearer {access_token}

步骤五：token 过期前用 refresh_token 刷新
```

### 1.4 核心 API 功能模块

```
Marketing API 通常包含以下模块：

账户管理
  ├── 获取授权账户列表
  ├── 查询账户余额
  └── 账户日预算设置

广告管理
  ├── 创建/修改/删除 Campaign（推广计划）
  ├── 创建/修改/删除 Ad Group（广告组）
  ├── 创建/修改/删除 Ad Creative（广告创意）
  ├── 广告状态管理（启用/暂停）
  └── 批量操作

素材管理
  ├── 上传图片/视频
  ├── 素材库查询
  └── 素材审核状态查询

数据报表
  ├── 账户级报表
  ├── 计划/组/创意级报表
  ├── 受众分析报表
  └── 实时数据查询

定向管理
  ├── 人群包上传/管理
  ├── 自定义受众创建
  └── Lookalike 人群扩展

转化追踪
  ├── 转化事件上报
  ├── 转化数据查询
  └── 归因数据
```

### 1.5 实战：创建一条广告（伪代码）

以下用伪代码展示通过 API 创建一条完整广告的流程：

```python
import requests

BASE_URL = "https://ad.oceanengine.com/open_api/v3.0"
HEADERS = {"Access-Token": "your_access_token"}

# 第一步：创建推广计划（Campaign）
campaign = requests.post(f"{BASE_URL}/campaign/create/", headers=HEADERS, json={
    "advertiser_id": 123456,
    "campaign_name": "春季跑鞋推广",
    "campaign_type": "FEED",           # 信息流
    "landing_type": "LINK",            # 落地页
    "budget_mode": "BUDGET_MODE_DAY",  # 日预算
    "budget": 10000                    # 100元（单位：分）
})
campaign_id = campaign.json()["data"]["campaign_id"]

# 第二步：创建广告组（Ad Group）
ad_group = requests.post(f"{BASE_URL}/ad/create/", headers=HEADERS, json={
    "advertiser_id": 123456,
    "campaign_id": campaign_id,
    "ad_name": "运动人群-一线城市",
    "delivery_range": "DEFAULT",
    "audience": {
        "city": [110000, 310000, 440100],  # 北京、上海、广州
        "age": ["AGE_18_23", "AGE_24_30"],
        "gender": "MALE",
        "interest_categories": [10001, 10002]  # 运动健身
    },
    "bid_type": "OCPM",
    "bid": 5000,                        # 目标转化成本 50元
    "convert_type": "CONVERT_TYPE_PAY", # 优化目标：付费
    "schedule_type": "SCHEDULE_FROM_NOW",
    "budget": 5000                      # 50元日预算
})
ad_id = ad_group.json()["data"]["ad_id"]

# 第三步：上传素材
with open("ad_video.mp4", "rb") as f:
    upload_resp = requests.post(
        f"{BASE_URL}/file/video/ad/",
        headers=HEADERS,
        data={"advertiser_id": 123456},
        files={"video_file": f}
    )
video_id = upload_resp.json()["data"]["video_id"]

# 第四步：创建广告创意
creative = requests.post(f"{BASE_URL}/creative/create/", headers=HEADERS, json={
    "advertiser_id": 123456,
    "ad_id": ad_id,
    "creative_material_mode": "CUSTOM_MATERIAL",
    "creatives": [{
        "title": "新款跑鞋限时特惠，立减200元",
        "video_id": video_id,
        "call_to_action_text": "立即购买",
        "landing_page_url": "https://example.com/shoes"
    }]
})
```

### 1.6 实战：拉取数据报表

```python
# 拉取广告组维度的日报表
report = requests.get(f"{BASE_URL}/report/ad/get/", headers=HEADERS, params={
    "advertiser_id": 123456,
    "start_date": "2026-04-20",
    "end_date": "2026-04-20",
    "fields": [
        "ad_name", "show_cnt", "click_cnt", "ctr",
        "convert_cnt", "convert_cost", "cost"
    ],
    "page": 1,
    "page_size": 20
})

# 响应示例
{
    "code": 0,
    "data": {
        "list": [
            {
                "ad_name": "运动人群-一线城市",
                "show_cnt": 150000,
                "click_cnt": 4500,
                "ctr": 3.0,
                "convert_cnt": 135,
                "convert_cost": 48.5,
                "cost": 6547.5
            }
        ],
        "page_info": {"total_number": 1, "page": 1}
    }
}
```

### 1.7 API 对接的常见坑

```
1. 频率限制（Rate Limiting）
   大多数平台有 QPS 限制（如 10-100 QPS）
   批量操作时需要控制请求速率
   建议：使用队列 + 限流器

2. 数据延迟
   报表数据通常有 5-30 分钟延迟
   转化数据可能有数小时延迟（归因窗口）
   不要用 API 数据做实时决策

3. 字段变更
   平台 API 会不定期更新，字段可能新增或废弃
   建议：做好版本管理，关注平台更新公告

4. 审核状态
   创建的广告不会立即生效，需要等审核通过
   需要轮询审核状态或监听回调通知

5. 金额单位
   不同平台的金额单位不同：
   巨量引擎：分（1元 = 100）
   腾讯广告：分
   Google Ads：微元（1元 = 1,000,000）
   务必注意单位转换！

6. 时区问题
   报表数据的时区要与业务时区对齐
   跨国投放时尤其需要注意
```

---

## 2. 媒体侧：广告 SDK 接入

### 2.1 接入流程概览

```
步骤一：注册媒体账号
  → 在广告平台（如穿山甲、腾讯优量汇）注册
  → 创建应用，获取 App ID
  → 创建广告位（Ad Slot），获取广告位 ID

步骤二：集成 SDK
  → 下载 SDK（Android/iOS）
  → 添加依赖到项目中
  → 配置权限和初始化代码

步骤三：实现广告位
  → 在合适的位置请求和展示广告
  → 处理广告事件回调（展示、点击、关闭等）

步骤四：测试与上线
  → 使用测试广告位验证集成
  → 提交审核
  → 上线后监控 eCPM 和填充率
```

### 2.2 Android SDK 集成示例（伪代码）

```kotlin
// build.gradle 添加依赖
dependencies {
    implementation "com.example.ad:sdk:5.0.0"
}

// Application 中初始化
class MyApp : Application() {
    override fun onCreate() {
        super.onCreate()
        AdSdk.init(this, AdConfig.Builder()
            .appId("your_app_id")
            .debug(BuildConfig.DEBUG)
            .build()
        )
    }
}

// 信息流广告加载与展示
class FeedAdLoader(private val context: Context) {
    
    fun loadAd(slotId: String) {
        val request = AdRequest.Builder()
            .slotId(slotId)
            .adCount(3)  // 请求 3 条广告
            .build()
        
        AdSdk.loadFeedAd(request, object : FeedAdListener {
            override fun onAdLoaded(ads: List<FeedAd>) {
                // 广告加载成功，插入到信息流列表中
                ads.forEach { ad ->
                    feedList.add(AdItem(ad))
                    adapter.notifyDataSetChanged()
                }
            }
            
            override fun onAdFailed(code: Int, msg: String) {
                Log.e("Ad", "加载失败: $code - $msg")
            }
            
            override fun onAdShow(ad: FeedAd) {
                // 广告展示回调（SDK 自动上报）
            }
            
            override fun onAdClick(ad: FeedAd) {
                // 广告点击回调
            }
        })
    }
}

// 激励视频广告
class RewardVideoLoader(private val activity: Activity) {
    
    fun loadAndShow(slotId: String) {
        val request = AdRequest.Builder().slotId(slotId).build()
        
        AdSdk.loadRewardVideo(request, object : RewardVideoListener {
            override fun onAdLoaded(ad: RewardVideoAd) {
                ad.show(activity)
            }
            
            override fun onRewardVerify(rewardType: String, amount: Int) {
                // 用户看完视频，发放奖励
                grantReward(rewardType, amount)
            }
            
            override fun onAdClosed() {
                // 广告关闭，可以预加载下一条
            }
        })
    }
}
```

### 2.3 广告变现优化策略

```
1. 瀑布流（Waterfall）
   按优先级依次请求多个广告源：
   广告源A（eCPM 高）→ 无填充 → 广告源B → 无填充 → 广告源C
   
   问题：优先级是预设的，不能动态调整

2. Header Bidding（头部竞价）
   同时向多个广告源发起请求，取最高价：
   广告源A: 30元 eCPM ─┐
   广告源B: 45元 eCPM ─┼→ 选择 B（最高价）
   广告源C: 25元 eCPM ─┘
   
   优势：每次展示都能获得最高收益

3. 混合模式
   自有广告系统 + 第三方广告源
   自有广告优先（利润高），填充不足时用第三方补充
```

### 2.4 Server-to-Server 对接（大型媒体）

```
适用于有自己广告系统的大型媒体，不使用 SDK，而是服务端直接对接。

请求流程：
  媒体服务器 → POST /ad/request → 广告平台
  
请求体示例：
{
    "slot_id": "ad_slot_001",
    "device": {
        "os": "android",
        "os_version": "13",
        "model": "Pixel 7",
        "device_id": "gaid_xxx"
    },
    "user": {
        "age": 25,
        "gender": "male"
    },
    "context": {
        "content_category": "sports",
        "network": "wifi"
    }
}

响应体示例：
{
    "ad_id": "ad_12345",
    "creative": {
        "type": "video",
        "url": "https://cdn.example.com/ad_video.mp4",
        "title": "新款跑鞋限时特惠",
        "description": "立减200元，仅限今天",
        "landing_url": "https://example.com/shoes"
    },
    "tracking": {
        "impression_url": "https://track.example.com/imp?id=xxx",
        "click_url": "https://track.example.com/click?id=xxx"
    },
    "price": 35.5  // eCPM
}

媒体需要：
  1. 渲染广告素材
  2. 广告展示时请求 impression_url（展示上报）
  3. 用户点击时请求 click_url（点击上报）
```

---

## 3. 转化追踪对接

### 3.1 为什么需要转化追踪？

```
广告平台需要知道广告带来了多少转化（购买、注册、下载等），用于：
  1. 广告主看效果报表
  2. oCPX 模型优化（没有转化数据，智能出价无法工作）
  3. 计费（CPA 模式下按转化计费）
```

### 3.2 转化追踪的实现方式

```
方式一：JS SDK / Pixel（网页端）
  在落地页嵌入追踪代码：
  
  <!-- 页面加载时上报 PV -->
  <script src="https://track.example.com/pixel.js"></script>
  <script>
    AdTracker.init("pixel_id_xxx");
    AdTracker.track("PageView");
  </script>
  
  <!-- 用户完成购买时上报转化 -->
  <script>
    AdTracker.track("Purchase", {
      value: 299,
      currency: "CNY",
      order_id: "order_123"
    });
  </script>

方式二：服务端回传（Server-to-Server）
  广告主服务器直接调用平台 API 上报转化：
  
  POST /conversion/report
  {
    "event_type": "purchase",
    "click_id": "click_id_from_landing_url",
    "timestamp": 1713686400,
    "value": 299,
    "order_id": "order_123"
  }
  
  优势：数据更准确，不受浏览器限制
  趋势：随着隐私政策收紧，服务端回传越来越重要

方式三：App SDK 自动上报
  集成广告平台的 SDK 后，SDK 自动追踪：
  - App 安装/激活
  - 注册
  - 付费
  - 自定义事件
```

### 3.3 归因对接

```
App 安装归因通常需要第三方归因平台：

主流归因平台：
  - AppsFlyer
  - Adjust
  - Branch
  - 热云数据（国内）

对接流程：
  1. 在归因平台创建应用
  2. 配置广告平台的 Postback URL
  3. 用户点击广告 → 归因平台记录点击
  4. 用户安装 App → 归因平台匹配点击 → 确认归因
  5. 归因平台回传数据给广告平台

归因匹配方式：
  - 设备 ID 匹配（GAID/IDFA）：最准确
  - 概率匹配（IP + UA + 时间）：IDFA 受限后的替代方案
  - 深度链接（Deferred Deep Link）：安装后直接跳转到指定页面
```

---

## 4. 完整对接架构示例

### 4.1 广告主自动化投放系统

```
┌─────────────────────────────────────────────────────┐
│              广告主自动化投放系统                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ 业务系统  │───▶│  投放决策引擎  │───▶│  API 调用  │ │
│  │ (商品/库存)│   │              │    │  适配层    │ │
│  └──────────┘    └──────────────┘    └─────┬─────┘ │
│                                            │       │
│                                     ┌──────┴──────┐│
│                                     │             ││
│                              ┌──────┴──┐  ┌──────┴──┐
│                              │巨量引擎  │  │腾讯广告  │
│                              │  API    │  │  API    │
│                              └─────────┘  └─────────┘
│                                                     │
│  ┌──────────┐    ┌──────────────┐                   │
│  │ 数据报表  │◀───│  数据同步服务  │◀── 定时拉取报表  │
│  │ 看板     │    │              │                   │
│  └──────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────┘

核心逻辑：
  1. 商品上新 → 自动创建广告计划
  2. 库存不足 → 自动暂停广告
  3. ROI 低于阈值 → 自动降低预算
  4. 爆款商品 → 自动加大投放
  5. 定时拉取报表 → 生成分析看板
```

### 4.2 媒体广告变现系统

```
┌─────────────────────────────────────────────────────┐
│              媒体广告变现系统                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  App 端                                             │
│  ┌──────────┐    ┌──────────────┐                   │
│  │ 广告位管理 │───▶│  广告请求调度  │                   │
│  │          │    │  (Mediation)  │                   │
│  └──────────┘    └──────┬───────┘                   │
│                         │                           │
│              ┌──────────┼──────────┐                │
│              ▼          ▼          ▼                │
│         ┌────────┐ ┌────────┐ ┌────────┐           │
│         │穿山甲   │ │优量汇   │ │自有广告 │           │
│         │SDK     │ │SDK     │ │系统     │           │
│         └────────┘ └────────┘ └────────┘           │
│                                                     │
│  服务端                                              │
│  ┌──────────┐    ┌──────────────┐                   │
│  │ 收入报表  │◀───│  数据聚合服务  │◀── 各平台数据     │
│  │ 分析     │    │              │                   │
│  └──────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────┘
```

---

## 5. 对接最佳实践

### 5.1 技术层面

```
1. API 调用封装
   - 统一的 HTTP 客户端，内置重试、超时、限流
   - 统一的错误处理和日志记录
   - 多平台适配层，屏蔽各平台 API 差异

2. 数据同步策略
   - 报表数据：每 15-30 分钟增量同步
   - 广告状态：每 5 分钟同步一次
   - 使用消息队列解耦同步任务

3. 异常处理
   - API 返回错误码分类处理
   - 审核拒绝自动通知
   - 预算耗尽自动告警
   - 账户异常（封禁等）紧急通知

4. 安全
   - Token 加密存储，不要硬编码
   - 使用 HTTPS
   - 定期轮换 refresh_token
```

### 5.2 业务层面

```
1. 从小规模开始
   先用少量预算测试 API 对接是否正确
   确认数据一致后再放量

2. 数据对账
   定期对比 API 拉取的数据与平台后台数据
   差异超过阈值时告警

3. 灰度发布
   新的自动化策略先在小范围测试
   效果验证后再全量推广

4. 监控告警
   - 广告消耗异常（突增/突降）
   - CTR/CVR 异常波动
   - API 调用失败率
   - 账户余额不足
```

---

## 6. 常见问题 FAQ

```
Q: API 创建的广告和在后台创建的有区别吗？
A: 没有区别，API 和后台操作的是同一套数据。

Q: 可以通过 API 获取竞争对手的数据吗？
A: 不可以，API 只能访问自己账户的数据。

Q: API 有调用次数限制吗？
A: 有，各平台不同。通常 10-100 QPS，超过会被限流。
   批量操作建议使用平台提供的批量接口。

Q: 转化数据回传有延迟怎么办？
A: 大多数平台支持延迟回传（24-72小时内），
   oCPX 模型会根据延迟转化数据自动校准。

Q: 多个平台的 API 差异大吗？
A: 核心概念类似，但字段命名、数据格式、错误码都不同。
   建议封装统一的适配层。

Q: SDK 接入会影响 App 性能吗？
A: 会有一定影响（包体积增加 1-5MB，内存占用增加）。
   建议：懒加载、按需初始化、监控性能指标。
```

---

## 小结

对接广告平台主要有两个方向：

1. **广告主侧（Marketing API）**：通过 API 自动化管理广告投放，核心是 OAuth 认证 + CRUD 操作 + 报表拉取
2. **媒体侧（SDK/API）**：通过 SDK 接入广告变现，核心是广告加载展示 + 事件上报 + 多广告源调度

关键注意事项：
- 认证用 OAuth 2.0，注意 token 刷新
- 注意各平台的金额单位和数据延迟
- 做好异常处理、数据对账和监控告警
- 转化追踪是 oCPX 的基础，务必正确对接

下一篇也是最后一篇，我们展望广告平台的未来——AI、隐私计算和行业趋势。

---

> 📖 系列目录
> 1. [广告平台的前世今生](./01-history.md)
> 2. [广告平台解决了什么问题](./02-problems-solved.md)
> 3. [广告平台核心概念全解析](./03-core-concepts.md)
> 4. [广告平台架构设计](./04-architecture.md)
> 5. [广告平台关键技术](./05-key-technologies.md)
> 6. **如何对接广告平台** ← 当前
> 7. 广告平台的未来
