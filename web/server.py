from flask import Flask, render_template, request
import dbconn as db

app = Flask(__name__, static_folder='static')

@app.route("/", methods=['POST', 'GET'])
def main():
  return render_template("index.html")

@app.route("/video1", methods=['POST', 'GET'])
def showVideo():
  if request.method == 'GET':
    face = db.database().getFace()
    video = db.database().getVideo()
    print("\nface : " + str(face))
    print("video : " + str(video) + "\n")
    return render_template("video1.html", face=face, video=video)

if __name__=="__main__":
  app.run(host='0.0.0.0', debug=True)