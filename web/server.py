from flask import Flask, render_template, request, flash, redirect, url_for
from dbconn import database as DB
from findSiamese import find
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User
from flask_login import login_user, LoginManager

app = Flask(__name__, static_folder='static')

app.config['SECRET_KEY'] = 'ssdf23fd 4fs d4'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db.init_app(app)

# flask-login 적용
login_manager = LoginManager()
login_manager.login_view = 'sign_in'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
  return User.query.get(id)

@app.route('/', methods=['GET', 'POST'])
def sign_in():
    if request.method == "POST":
        email = request.form.get('email')
        password1 = request.form.get('password1')
        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password1):
                flash('로그인 완료', category='success')
                login_user(user, remember=True)
                return redirect(url_for('video'))
            else:
                flash('비밀번호가 다릅니다.', category='error')
        else:
            flash('해당 이메일 정보가 없습니다.', category='error')
    return render_template('sign_in.html', user="SemiCircle")


@app.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == "POST":
        # Check Request data
        data = request.form
        # Split Data
        email = request.form.get('email')
        name = request.form.get('name')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        group = request.form.get('group')
        # 유효성 검사
        user = User.query.filter_by(email=email).first()
        if user:
            flash("이미 가입된 이메일입니다.", category='error')
        elif len(email) < 5:
            flash("이메일은 5자 이상입니다.", category="error")
        elif len(name) < 2:
            flash("이름은 2자 이상입니다.", category="error")
        elif password1 != password2:
            flash('비밀번호가 서로 다릅니다', category='error')
        elif len(password1) < 4:
            flash("비밀번호가 너무 짧습니다.", category="error")
        else:
            new_user = User(email=email,
                            name=name,
                            password=generate_password_hash(password1, method='sha256'), group=group)
            db.session.add(new_user)
            db.session.commit()
            flash("회원가입 완료.", category="success")  # Create User -> DB
            # return render_template('sign_in.html')
            return redirect(url_for('sign_in'))
    return render_template('sign_up.html')


@app.route("/video", methods=['POST', 'GET'])
def video():
  return render_template("index.html")

@app.route("/video/1", methods=['POST', 'GET'])
def showVideo():
  if request.method == 'GET':
    face = DB().getFace()
    video = DB().getVideo()
    error = {}

    if face:
      if video: # 탐지된 face, video가 둘다 존재하는 경우
        print("\nface : " + str(face))
        print("video : " + str(video) + "\n")
        face_name = find.similar(face)
        print("face name : " + str(face_name))
        result = ((f, n) for f, n in zip(face, face_name))
        return render_template("video1.html", video=video, result=result)
      else: # face만 존재하는 경우
        error["face"] = True
        error["video"] = False
    else:
      if video: # video만 존재하는 경우
        error["face"] = False
        error["video"] = True
      else: # 탐지된 face, video 둘다 존재하지 않는 경우
        error["face"] = False
        error["video"] = False
    
    if not error:
      return render_template("failDetect.html", video=video, error=error)
    
@app.route("/view/chart", methods=['POST', 'GET'])
def viewChart():
  return render_template("visualization.html")

if __name__=="__main__":
  app.run(host='0.0.0.0', debug=True)