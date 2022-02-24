from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError

allowed_extn = ['jpg','png','jpeg']


class ImageCaptionForm(FlaskForm):
    picture = FileField('Upload Picture', validators=[DataRequired(), FileAllowed(allowed_extn)])
    submit = SubmitField('Submit')