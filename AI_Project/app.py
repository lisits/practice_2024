from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = Flask(__name__)

# Конфигурация базы данных
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost/AI_project'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Инициализация базы данных
db = SQLAlchemy(app)

# Модель пользователя
class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    bio = db.Column(db.String(200), nullable=False)
    avatar = db.Column(db.String(200), nullable=False)

    def __init__(self, name, bio, avatar):
        self.name = name
        self.bio = bio
        self.avatar = avatar

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        bio = request.form['bio']
        avatar = pipe(bio).images[0]
        avatar_path = f'static/avatars/{name}.png'
        avatar.save(avatar_path)
        new_user = Person(name=name, bio=bio, avatar=avatar_path)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('profile', username=name))
    return render_template('register.html')

@app.route('/profile/<username>')
def profile(username):
    user = Person.query.filter_by(name=username).first()
    return render_template('profile.html', user=user)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
