from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, StringField, SubmitField, FloatField
from wtforms.validators import DataRequired, NumberRange


class ClassifierForm(FlaskForm):
    classifier = SelectField('Escolha o Classificador:', choices=[
        ('KNN', 'KNN'),
        ('SVM', 'SVM'),
        ('MLP', 'MLP'),
        ('DT', 'Decision Tree'),
        ('RF', 'Random Forest')
    ])
    submit = SubmitField('Treinar')


class KNNForm(FlaskForm):
    n_neighbors = IntegerField('Número de Vizinhos', validators=[DataRequired(), NumberRange(min=1)])
    submit = SubmitField('Treinar KNN')

class SVMForm(FlaskForm):
    # C = FloatField('C (Regularização)', validators=[DataRequired()])
    kernel = SelectField('Kernel', choices=[('linear', 'Linear'), ('rbf', 'RBF'), ('poly', 'Polinomial')])
    degree = IntegerField('Grau (para kernel polinomial)', validators=[NumberRange(min=1)], default=3)
    submit = SubmitField('Treinar SVM')

class MLPForm(FlaskForm):
    hidden_layer_sizes = StringField('Tamanhos das Camadas Ocultas (e.g. 10,10)', validators=[DataRequired()])
    max_iter = IntegerField('Número Máximo de Iterações', validators=[DataRequired(), NumberRange(min=1)])
    submit = SubmitField('Treinar MLP')

class DTForm(FlaskForm):
    max_depth = IntegerField('Profundidade Máxima', validators=[NumberRange(min=1)])
    submit = SubmitField('Treinar DT')

class RFForm(FlaskForm):
    n_estimators = IntegerField('Número de Árvores', validators=[DataRequired(), NumberRange(min=1)])
    max_depth = IntegerField('Profundidade Máxima', validators=[NumberRange(min=1)], default=None)
    submit = SubmitField('Treinar RF')







