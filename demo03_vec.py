"""
demo03_vec.py  定义一种投资策略  判断是否可以实施
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

def dmy2ymd(dmy):
	# 把日月年字符串转为年月日字符串
	dmy = str(dmy, encoding='utf-8')
	d = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
	ymd = d.strftime('%Y-%m-%d')
	return ymd


# 加载文件
dates, opening_prices, highest_prices, \
lowest_prices, closing_prices = \
    np.loadtxt('../da_data/aapl.csv',
	delimiter=',', usecols=(1, 3, 4, 5, 6),
	unpack=True, dtype='M8[D], f8, f8, f8, f8',
	converters={1:dmy2ymd})

# 绘制收盘价的折线图
mp.figure('AAPL', facecolor='lightgray')
mp.title('AAPL', fontsize=18)
mp.xlabel('Date', fontsize=14)
mp.ylabel('Price', fontsize=14)
mp.grid(linestyle=':')
mp.tick_params(labelsize=10)
# 设置刻度定位器
ax = mp.gca()
#每周一一个主刻度
maloc = md.WeekdayLocator(byweekday=md.MO)
ax.xaxis.set_major_locator(maloc)
# 设置主刻度日期的格式
ax.xaxis.set_major_formatter(
	md.DateFormatter('%Y-%m-%d'))
#DayLocator:每天一个次刻度
ax.xaxis.set_minor_locator(md.DayLocator())
# 把dates的数据类型改为matplotlib的日期类型
dates = dates.astype(md.datetime.datetime)


def profit(oprice, hprice, lprice, cprice):
	# 定义买入卖出策略函数
	buy_price = oprice * 0.99
	if lprice <= buy_price <= hprice:
		return (cprice-buy_price)/buy_price
	else:
		return np.nan

# 计算使用该策略后，30天中每天的收益
profits=np.vectorize(profit)(opening_prices, 
	highest_prices, lowest_prices, 
	closing_prices)
print(profits)
#  获取profits中isnan的掩码数组
nan_mask = np.isnan(profits)
dates = dates[~nan_mask]
profits = profits[~nan_mask]
# 绘制收益线
mp.plot(dates, profits, 'o-', label='profit')
print(np.mean(profits))

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()





