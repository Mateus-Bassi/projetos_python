from flask import flash, redirect, render_template, request, url_for
from sklearn.model_selection import train_test_split
from app import app
from .forms import ClassifierForm, KNNForm, SVMForm, MLPForm, DTForm, RFForm
from ml import carregar_dados, treinar_e_avaliar

@app.route('/select_classifier', methods=['GET', 'POST'])
def select_classifier():
    classifier = request.args.get('classifier', 'KNN')  # KNN como default se nenhum classificador for passado
    form = None
    parametros = {}
    
    if classifier == 'KNN':
        form = KNNForm(request.form)
        if form.validate_on_submit():
            parametros = {'n_neighbors': form.n_neighbors.data}
    elif classifier == 'SVM':
        form = SVMForm(request.form)
        if form.validate_on_submit():
            parametros = {'kernel': form.kernel.data, 'degree': form.degree.data}
    elif classifier == 'MLP':
        form = MLPForm(request.form)
        if form.validate_on_submit():
            hidden_layer_sizes = tuple(map(int, form.hidden_layer_sizes.data.split(',')))
            parametros = {'hidden_layer_sizes': hidden_layer_sizes, 'max_iter': form.max_iter.data}
    elif classifier == 'DT':
        form = DTForm(request.form)
        if form.validate_on_submit():
            parametros = {'max_depth': form.max_depth.data}
    elif classifier == 'RF':
        form = RFForm(request.form)
        if form.validate_on_submit():
            parametros = {'n_estimators': form.n_estimators.data, 'max_depth': form.max_depth.data}

    if request.method == 'POST' and form.validate():
        X, y = carregar_dados()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        caminho_imagem = treinar_e_avaliar(classifier, parametros, X_train, y_train, X_test, y_test)

        flash('Modelo treinado com sucesso!')
        return render_template('classifier_form.html', form=form, classifier=classifier, matrix_image=caminho_imagem)

    return render_template('classifier_form.html', form=form, classifier=classifier)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ClassifierForm()
    if form.validate_on_submit():
        classificador_escolhido = form.classifier.data
        return redirect(url_for('select_classifier', classifier=classificador_escolhido))

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
