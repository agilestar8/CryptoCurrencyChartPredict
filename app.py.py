from flask import Flask,render_template,request
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.resources import INLINE
import pybithumb
import pyupbit
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import warnings
# from bokeh.util.string import encode_utf8 --> import오류로 .encode(encoding = 'UTF-8')로 대체
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)

app = Flask(__name__)

# 홈
@app.route("/",methods = [ "GET", "POST"])
def index():
    return render_template('index.html')

@app.route('/post', methods = ["GET","POST"])
def chart_page():
    if request.method == "POST":
        pagename2 = request.form['chart']
        try:
            df = pybithumb.get_candlestick(pagename2)
            df = df.reset_index()
            df.rename(columns={'time': 'date'}, inplace=True)
            df["ma5"] = df["close"].rolling(5).mean()
            df["ma10"] = df["close"].rolling(10).mean()
            df["ma30"] = df["close"].rolling(30).mean()

            # 1.VWAP
            v = df['volume'].values
            tp = (df['low'] + df['close'] + df['high']).div(3).values
            df = df.assign(vwap=(tp * v).cumsum() / v.cumsum())
            df["diff"] = df["close"] - df["vwap"]  # +면 상승세, -면 하락세

            # 2.RSI
            df["U"] = np.where(df["close"].diff(1) > 0, df["close"].diff(1), 0)  # 전일대비 상승분 가격
            df["D"] = np.where(df["close"].diff(1) < 0, df["close"].diff(1) * (-1), 0)  # 전일대비 하락분 가격

            period = 14  # 하락분의 14일 평균을 구해줍니다.
            df["AU"] = df["U"].rolling(window=period, min_periods=period).mean()  # 14일간 전일대비 평균 상승가격
            df["AD"] = df["D"].rolling(window=period, min_periods=period).mean()  # 14일간 전일대비 평균 하락가격

            # RSI = AU / (AU + AD) 의 백분율을 RSI 로 계산해줍니다.
            RSI = (df["AU"] / (df["AU"] + df["AD"])) * 100
            df["rsi"] = RSI

            # 3.MACD
            def fnMACD(m_Df, m_NumFast=12, m_NumSlow=26, m_NumSignal=9):
                m_Df['EMAFast'] = m_Df['close'].ewm(span=m_NumFast, min_periods=m_NumFast - 1).mean()  # 12일의 지수평균이동
                m_Df['EMASlow'] = m_Df['close'].ewm(span=m_NumSlow, min_periods=m_NumSlow - 1).mean()  # 26
                m_Df['MACD'] = m_Df['EMAFast'] - m_Df['EMASlow']
                m_Df['MACDSignal'] = m_Df['MACD'].ewm(span=m_NumSignal, min_periods=m_NumSignal - 1).mean()  # 9
                m_Df['MACDDiff'] = m_Df['MACD'] - m_Df['MACDSignal']
                return m_Df

            df = fnMACD(df)

            # 4.BolingerBand
            def fnBolingerBand(m_DF, n=20, k=2):
                m_DF['ma20'] = m_DF['close'].rolling(n).mean()
                m_DF['upper'] = m_DF['close'].rolling(n).mean() + k * m_DF['close'].rolling(n).std()
                m_DF['lower'] = m_DF['close'].rolling(n).mean() - k * m_DF['close'].rolling(n).std()
                return m_DF

            df = fnBolingerBand(df)

            # 최근 1년간의 데이터의 동향 파악
            df2 = df.iloc[-360:]
            inc = df2.close >= df2.open  # 시세가 증가하는 인덱스
            dec = df2.open > df2.close  # 시세가 감소하는 인덱스

            거래량 = int(df["volume"].iloc[-1])
            고가 = int(df["high"].iloc[-1])
            저가 = int(df["low"].iloc[-1])
            현재가격 = int(df["close"].iloc[-1])
            전일대비 = int(df["close"].iloc[-1] - df["close"].iloc[-2])
            전일퍼센트 = round((df["close"].iloc[-1] - df["close"].iloc[-2])/df["close"].iloc[-1]*100,2)

            js_resources = INLINE.render_js()
            css_resources = INLINE.render_css()

            ## candle stick

            major_label = {
                i: date.strftime('%Y-%m-%d') for i, date in enumerate(pd.to_datetime(df["date"]))
            }

            major_label.update({len(df2): ""})

            candlechart = figure(plot_width=900,
                                 plot_height=350,
                                 x_range = (len(df)-150, len(df)),
                                 # y_range=(30000000, max(df2["close"])+2000000),
                                 tools=['xpan, crosshair, xwheel_zoom, reset, hover, box_select, save'])

            candlechart.segment(df2.index[inc], df2.high[inc], df2.index[inc], df2.low[inc], color="green")
            candlechart.segment(df2.index[dec], df2.high[dec], df2.index[dec], df2.low[dec], color="green")
            candlechart.vbar(df2.index[inc], 0.5, df2.open[inc], df2.close[inc], fill_color="blue", line_color="blue")
            candlechart.vbar(df2.index[dec], 0.5, df2.open[dec], df2.close[dec], fill_color="red", line_color="red")

            candlechart.line(df2.index, df2["vwap"], line_color="orange", legend_label="vwap")
            candlechart.line(df2.index, df2["rsi"], line_color="red", legend_label="RSI")
            candlechart.line(df2.index, df2["MACD"], line_color="blue", legend_label="MACD")

            candlechart.line(df2.index, df2["upper"], line_color="purple", legend_label="BolingerBand")
            candlechart.line(df2.index, df2["lower"], line_color="purple")

            candlechart.legend.location = "top_left"


            candlechart.xaxis.major_label_overrides = major_label
            candlechart.yaxis[0].formatter = NumeralTickFormatter(format="0.0")
            # candlechart.xaxis.axis_label = 'Date'
            candlechart.yaxis.axis_label = 'Price'
            candlechart.title.text = ''
            candlechart.title.text_font_size = '25px'
            candlechart.title.align = 'center'


            ## bar chart 거래량
            barchart = figure(plot_width=900, plot_height=150,
                              x_range = (len(df)-150, len(df)),
                              # y_range=(0, 15000),
                              tools=['xpan, crosshair, xwheel_zoom, reset, hover, box_select, save'])
            barchart.vbar(df2.index[dec], 0.5, df2[dec]["volume"], fill_color="blue", line_color="blue")
            barchart.vbar(df2.index[inc], 0.5, df2[inc]["volume"], fill_color="red", line_color="red")
            # barchart.legend.location = "top_left"

            barchart.xaxis.major_label_overrides = major_label
            barchart.yaxis[0].formatter = NumeralTickFormatter(format="0.0")
            barchart.xaxis.axis_label = 'Date'
            barchart.yaxis.axis_label = 'Volume'
            # barchart.title.text='Volume'
            barchart.title.text_font_size = '25px'
            barchart.title.align = 'center'

            kk = gridplot( [[candlechart],[barchart]] )
            script, div = components(kk)

            return render_template('chart_page.html',
                                   coin_name=pagename2,
                                   vol = 거래량,
                                   high = 고가,
                                   low = 저가,
                                   price = 현재가격,
                                   diff1day = 전일대비,
                                   dffper = 전일퍼센트,
                                   plot_script=script,
                                   plot_div=div,
                                   js_resources=js_resources,
                                   css_resources=css_resources,
                                   ).encode(encoding='UTF-8')
        except:
            return "잘못된 입력 또는 error 발생"


@app.route('/analysis', methods = ["GET","POST"])
def analysis_page():
    if request.method == "POST":
        coin_name = request.form["analysis"]
        try:
            # df = pybithumb.get_candlestick(coin_name)
            df = pyupbit.get_ohlcv(coin_name, interval="minute60", count=5000)

            # 1.VWAP
            v = df['volume'].values
            tp = (df['low'] + df['close'] + df['high']).div(3).values
            df = df.assign(vwap=(tp * v).cumsum() / v.cumsum())
            df["diff"] = df["close"] - df["vwap"]  # +면 상승세, -면 하락세

            # 2.RSI
            df["U"] = np.where(df["close"].diff(1) > 0, df["close"].diff(1), 0)  # 전일대비 상승분 가격
            df["D"] = np.where(df["close"].diff(1) < 0, df["close"].diff(1) * (-1), 0)  # 전일대비 하락분 가격

            period = 14  # 하락분의 14일 평균을 구해줍니다.
            df["AU"] = df["U"].rolling(window=period, min_periods=period).mean()  # 14일간 전일대비 평균 상승가격
            df["AD"] = df["D"].rolling(window=period, min_periods=period).mean()  # 14일간 전일대비 평균 하락가격

            # RSI = AU / (AU + AD) 의 백분율을 RSI 로 계산해줍니다.
            RSI = (df["AU"] / (df["AU"] + df["AD"])) * 100
            df["rsi"] = RSI

            # 모델 및 파라미터 설정
            model = ARIMA(df["close"], order=(0, 2, 1))
            model_fit = model.fit(trend='nc', full_output=True)

            # 24시간 예측
            pred = model_fit.forecast(steps=24)
            result = pred[-1]

            target_df = pyupbit.get_ohlcv(coin_name, interval="day", count=2)
            target_price = target_df.iloc[0]['close'] + (target_df.iloc[0]['high'] - df.iloc[0]['low']) * 0.5
            cur_price = pyupbit.get_current_price()

            invest_recommand = 0
            # 추천 프로세스
            if target_price < cur_price:
                invest_recommand += 1
                if result > cur_price:
                    invest_recommand += 1
                    if df["rsi"] <= 30:
                        invest_recommand += 1
                        if df["ris"] <= 20:
                            invest_recommand += 1

            if df["rsi"].iloc[-1] >= 70:
                invest_recommand -= 3
            elif df["rsi"].iloc[-1] >= 65:
                invest_recommand -= 2
            elif df["rsi"].iloc[-1] >= 60:
                invest_recommand -= 1

            vwap = df["vwap"].iloc[-1]
            rsi = df["rsi"].iloc[-1]

            return render_template('analysis.html',
                                   coin_name = coin_name,
                                   target_price = int(target_price),
                                   cur_price = int(cur_price),
                                   vwap = int(vwap),
                                   rsi = int(rsi),
                                   invest_recommand = invest_recommand
                                   )
        except:
            "분석 페이지 - 잘못된 입력 또는 error 발생"

## 동적 라우팅
@app.route('/<pagename2>')
def chart_page2(pagename2):
    try:
        df = pybithumb.get_candlestick(pagename2)
        df = df.reset_index()
        df.rename(columns={'time': 'date'}, inplace=True)
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma30"] = df["close"].rolling(30).mean()

        # 1.VWAP
        v = df['volume'].values
        tp = (df['low'] + df['close'] + df['high']).div(3).values
        df = df.assign(vwap=(tp * v).cumsum() / v.cumsum())
        df["diff"] = df["close"] - df["vwap"]  # +면 상승세, -면 하락세

        # 2.RSI
        df["U"] = np.where(df["close"].diff(1) > 0, df["close"].diff(1), 0)  # 전일대비 상승분 가격
        df["D"] = np.where(df["close"].diff(1) < 0, df["close"].diff(1) * (-1), 0)  # 전일대비 하락분 가격

        period = 14  # 하락분의 14일 평균을 구해줍니다.
        df["AU"] = df["U"].rolling(window=period, min_periods=period).mean()  # 14일간 전일대비 평균 상승가격
        df["AD"] = df["D"].rolling(window=period, min_periods=period).mean()  # 14일간 전일대비 평균 하락가격

        # RSI = AU / (AU + AD) 의 백분율을 RSI 로 계산해줍니다.
        RSI = (df["AU"] / (df["AU"] + df["AD"])) * 100
        df["rsi"] = RSI

        # 3.MACD
        def fnMACD(m_Df, m_NumFast=12, m_NumSlow=26, m_NumSignal=9):
            m_Df['EMAFast'] = m_Df['close'].ewm(span=m_NumFast, min_periods=m_NumFast - 1).mean()  # 12일의 지수평균이동
            m_Df['EMASlow'] = m_Df['close'].ewm(span=m_NumSlow, min_periods=m_NumSlow - 1).mean()  # 26
            m_Df['MACD'] = m_Df['EMAFast'] - m_Df['EMASlow']
            m_Df['MACDSignal'] = m_Df['MACD'].ewm(span=m_NumSignal, min_periods=m_NumSignal - 1).mean()  # 9
            m_Df['MACDDiff'] = m_Df['MACD'] - m_Df['MACDSignal']
            return m_Df

        df = fnMACD(df)

        # 4.BolingerBand
        def fnBolingerBand(m_DF, n=20, k=2):
            m_DF['ma20'] = m_DF['close'].rolling(n).mean()
            m_DF['upper'] = m_DF['close'].rolling(n).mean() + k * m_DF['close'].rolling(n).std()
            m_DF['lower'] = m_DF['close'].rolling(n).mean() - k * m_DF['close'].rolling(n).std()
            return m_DF

        df = fnBolingerBand(df)

        # 최근 1년간의 데이터의 동향 파악
        df2 = df.iloc[-360:]
        inc = df2.close >= df2.open  # 시세가 증가하는 인덱스
        dec = df2.open > df2.close  # 시세가 감소하는 인덱스

        거래량 = int(df["volume"].iloc[-1])
        고가 = int(df["high"].iloc[-1])
        저가 = int(df["low"].iloc[-1])
        현재가격 = int(df["close"].iloc[-1])
        전일대비 = int(df["close"].iloc[-1] - df["close"].iloc[-2])
        전일퍼센트 = round((df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-1] * 100, 2)

        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        ## candle stick

        major_label = {
            i: date.strftime('%Y-%m-%d') for i, date in enumerate(pd.to_datetime(df["date"]))
        }

        major_label.update({len(df2): ""})

        candlechart = figure(plot_width=900,
                             plot_height=350,
                             x_range=(len(df) - 150, len(df)),
                             # y_range=(30000000, max(df2["close"])+2000000),
                             tools=['xpan, crosshair, xwheel_zoom, reset, hover, box_select, save'])

        candlechart.segment(df2.index[inc], df2.high[inc], df2.index[inc], df2.low[inc], color="green")
        candlechart.segment(df2.index[dec], df2.high[dec], df2.index[dec], df2.low[dec], color="green")
        candlechart.vbar(df2.index[inc], 0.5, df2.open[inc], df2.close[inc], fill_color="blue", line_color="blue")
        candlechart.vbar(df2.index[dec], 0.5, df2.open[dec], df2.close[dec], fill_color="red", line_color="red")

        candlechart.line(df2.index, df2["vwap"], line_color="orange", legend_label="vwap")
        candlechart.line(df2.index, df2["rsi"], line_color="red", legend_label="RSI")
        candlechart.line(df2.index, df2["MACD"], line_color="blue", legend_label="MACD")

        candlechart.line(df2.index, df2["upper"], line_color="purple", legend_label="BolingerBand")
        candlechart.line(df2.index, df2["lower"], line_color="purple")

        candlechart.legend.location = "top_left"

        candlechart.xaxis.major_label_overrides = major_label
        candlechart.yaxis[0].formatter = NumeralTickFormatter(format="0.0")
        # candlechart.xaxis.axis_label = 'Date'
        candlechart.yaxis.axis_label = 'Price'
        candlechart.title.text = ''
        candlechart.title.text_font_size = '25px'
        candlechart.title.align = 'center'

        ## bar chart 거래량
        barchart = figure(plot_width=900, plot_height=150,
                          x_range=(len(df) - 150, len(df)),
                          # y_range=(0, 15000),
                          tools=['xpan, crosshair, xwheel_zoom, reset, hover, box_select, save'])
        barchart.vbar(df2.index[dec], 0.5, df2[dec]["volume"], fill_color="blue", line_color="blue")
        barchart.vbar(df2.index[inc], 0.5, df2[inc]["volume"], fill_color="red", line_color="red")
        # barchart.legend.location = "top_left"

        barchart.xaxis.major_label_overrides = major_label
        barchart.yaxis[0].formatter = NumeralTickFormatter(format="0.0")
        barchart.xaxis.axis_label = 'Date'
        barchart.yaxis.axis_label = 'Volume'
        # barchart.title.text='Volume'
        barchart.title.text_font_size = '25px'
        barchart.title.align = 'center'

        kk = gridplot([[candlechart], [barchart]])
        script, div = components(kk)

        return render_template('chart_page.html',
                               coin_name=pagename2,
                               vol=거래량,
                               high=고가,
                               low=저가,
                               price=현재가격,
                               diff1day=전일대비,
                               dffper=전일퍼센트,
                               plot_script=script,
                               plot_div=div,
                               js_resources=js_resources,
                               css_resources=css_resources,
                               ).encode(encoding='UTF-8')
    except:
        return "잘못된 입력 또는 error 발생"

if __name__ == '__main__': # 로컬호스팅
    app.run(debug = True)

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", debug=True) 

