
def config_arvore_decisao(root):
    from sklearn.tree import DecisionTreeClassifier

    opt_criterion = root.selectbox("Métrica", options=['gini','entropy'], index=0)

    model = DecisionTreeClassifier(criterion=opt_criterion)

    return model

def config_naive_bayes(root):
    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()

    return model

def config_regressao_logistica(root):
    from sklearn.linear_model import LogisticRegression

    opt_solver = root.selectbox("Solver", options=['liblinear','sag','saga'], index=0)

    model = LogisticRegression(solver=opt_solver)

    return model

def config_svm(root):
    from sklearn.svm import SVC

    opt_kernel = root.selectbox("Kernel", options=['linear','poly','sigmoid'], index=0)

    model = SVC(kernel=opt_kernel)

    return model


MODELS = {
    'Naive Bayes': config_naive_bayes,
    'SVM': config_svm,
    'Regressão Logistica': config_regressao_logistica,
    'Àrvore de Decisão': config_arvore_decisao
}
