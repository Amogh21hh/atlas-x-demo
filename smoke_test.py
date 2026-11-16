from streamlit.testing.v1 import AppTest
at = AppTest.from_file("streamlit_app.py", default_timeout=600).run()
assert not at.exception, at.exception
print("tabs:", [t.label for t in at.tabs])
print("planner header:", [h.value for h in at.subheader])
at.button[0].click().run()   # research lab
assert not at.exception, at.exception
print("success:", [s.value[:120] for s in at.success])
at.button[1].click().run()   # risk sandbox default: VUSA 1500 shares
assert not at.exception, at.exception
print("risk:", [e.value for e in at.error], [s.value[:80] for s in at.success][-1:])
