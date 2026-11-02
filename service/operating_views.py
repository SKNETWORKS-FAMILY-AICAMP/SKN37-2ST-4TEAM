"""기존 화면 디자인을 사용하는 대응 결과 입력·조회 화면."""
from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

import operations as O
import ui


def show_result(result):
    k = st.columns(4)
    difference = result['diff'] * 100
    with k[0]:
        ui.kpi('대응군 이탈률', f"{result['rate_t']:.1%}", f"{result['churn_t']} / {result['n_t']}명")
    with k[1]:
        ui.kpi('비교군 이탈률', f"{result['rate_c']:.1%}", f"{result['churn_c']} / {result['n_c']}명")
    with k[2]:
        ui.kpi('이탈률 감소', f'{difference:+.1f}%포인트', '양수이면 대응군의 이탈률이 낮음', hot=True)
    with k[3]:
        ui.kpi('추정 순이익', f"{result['estimated_profit']:,.0f}원", f"연락·혜택 비용 {result['total_cost']:,.0f}원 반영")
    if result['causal_support']:
        st.success('무작위 배정한 두 그룹에서 이탈 감소가 관찰됐습니다. 전체 효과는 결과 보고서에서 확인하세요.')
    elif difference > 0:
        st.info('대응군의 이탈률이 더 낮습니다. 현재 결과만으로 대응의 효과를 확정하지는 않습니다.')
    elif difference < 0:
        st.warning('대응군의 이탈률이 더 높습니다. 대상 선정과 제공한 혜택을 점검하세요.')
    else:
        st.info('두 그룹의 이탈률 차이가 없습니다.')
    fig = go.Figure(go.Bar(
        x=['대응군', '비교군'], y=[result['rate_t'], result['rate_c']],
        marker_color=[ui.MINT, ui.PINK],
        text=[f"{result['rate_t']:.1%}", f"{result['rate_c']:.1%}"], textposition='outside'))
    fig.update_layout(yaxis_title='14일 이탈률', yaxis_tickformat='.0%',
                      yaxis_range=[0, max(result['rate_t'], result['rate_c'], .01)*1.35])
    ui.plot(ui.style(fig, 260, legend=False))


def response_results(active_date):
    ui.section('대응 후 이탈이 줄었나요?', '연락·혜택을 제공한 고객과 같은 기간의 비교 고객 결과를 기록합니다.')
    saved = O.load_campaigns()
    if not saved:
        st.info('아직 기록한 대응 결과가 없습니다. 14일 관찰을 마친 캠페인부터 입력하세요.')
    else:
        selected = st.selectbox('확인할 캠페인', range(len(saved)),
                                format_func=lambda i: f"{saved[i]['name']} · {saved[i]['start']} ~ {saved[i]['end']}")
        show_result(saved[selected])
        st.dataframe(O.campaign_table(saved), hide_index=True)
        st.download_button('대응 결과 내려받기', O.campaign_table(saved).to_csv(index=False).encode('utf-8-sig'),
                           file_name='대응결과.csv', mime='text/csv')
    with st.expander('새 대응 결과 기록', expanded=not saved):
        with st.form('response_result_form'):
            a, b = st.columns(2)
            name = a.text_input('캠페인 이름', placeholder='예: 이용 감소 고객 맞춤 추천')
            start = b.date_input('대응 시작일', value=active_date)
            end = start + timedelta(days=14)
            st.caption(f'결과 확인일: {end:%Y-%m-%d} · 실제 관찰한 고객 수만 입력하세요.')
            a, b, c, d = st.columns(4)
            n_t = a.number_input('대응군 고객 수', min_value=0, value=0, step=1)
            ch_t = b.number_input('대응군 이탈 고객 수', min_value=0, value=0, step=1)
            n_c = c.number_input('비교군 고객 수', min_value=0, value=0, step=1)
            ch_c = d.number_input('비교군 이탈 고객 수', min_value=0, value=0, step=1)
            a, b = st.columns(2)
            value = a.number_input('유지 고객 한 명의 추정 가치 (원)', min_value=0, value=0, step=1000)
            cost = b.number_input('대응 고객 한 명당 실제 비용 (원)', min_value=0, value=0, step=100)
            randomized = st.checkbox('대응 전에 고객을 두 그룹으로 무작위 배정했습니다.')
            completed = st.checkbox('두 그룹 모두 같은 14일 관찰을 완료했습니다.')
            submit = st.form_submit_button('결과 계산하고 저장', type='primary')
        if submit:
            try:
                result = O.campaign_result(name, start, n_t, ch_t, n_c, ch_c, value, cost,
                                           randomized=randomized, completed=completed)
                O.save_campaign(result)
            except (ValueError, OSError) as error:
                st.error(str(error))
            else:
                st.session_state.response_saved = '대응 결과를 저장했습니다.'
                st.rerun()
    if 'response_saved' in st.session_state:
        st.success(st.session_state.pop('response_saved'))
