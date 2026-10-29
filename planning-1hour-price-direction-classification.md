# Market data APIs for 1-hour price direction classification

## Recommendation

Use `yfinance` for a personal script, research, or an early prototype. It is free to use and simple from Python, but it relies on Yahoo Finance data rather than an official Yahoo API. Data freshness, rate limits, uptime, and commercial-use rights are less clearly guaranteed.

For a production application or a workflow that needs documented API access, consider a market-data provider such as Twelve Data. It offers forex and equity data through documented endpoints with API-key access. Check that the chosen plan covers the specific instrument, interval, freshness, request volume, and usage rights you need.

## Quick comparison

| | `yfinance` | Twelve Data |
|---|---|---|
| Cost | Free to use, subject to Yahoo access limits and terms | Free tier and paid plans |
| Setup | Very easy in Python | REST API or SDK; API key required |
| Forex | Available through Yahoo Finance symbols | Dedicated forex endpoints |
| S&P 500 | Index ticker such as `^GSPC`; `SPY` is the ETF | Verify the index symbol and plan coverage |
| Best fit | Personal analysis, scripts, prototypes | Applications needing documented API access |

## For one-hour direction labels

Both could supply hourly historical bars for prototyping a classifier. Before building around either source, verify hourly-bar availability and history depth for the exact forex pairs and S&P instrument, along with timezone conventions, missing-bar handling, rate limits, and licensing. Use a consistent definition of the target, for example whether the next hourly close is above or below the current close.

**Sources:** [Twelve Data documentation](https://twelvedata.com/docs) · [Alpha Vantage documentation](https://www.alphavantage.co/documentation/)
