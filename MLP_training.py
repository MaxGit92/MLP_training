import tensorflow as tf
import numpy as np
import pandas as pd
import shelve
import time
import os
import matplotlib.pyplot as plt
from sklearn import utils

class MLP_training:
    """Classe utilisant la bibliothèque TensorFlow, permettant 
    d'entraîner un réseau de neurones, mais aussi de trouver 
    la structure la plus intéressante possible pour un certain 
    jeu de données.
    Effectue en quelque sorte un gridsearch pour trouver une
    structure performante."""
    
    """
    Fonction d'initialisation de l'objet. Les paramètres sont ceux qui ne changerons pas 
    entre 2 réseaux de neurones différents et ceux qui sont le résultat de notre 
    entraînement de structure.
    
    output_dim : int, (default=1), est la dimension de sortie du réseau (correspond au
        nombre de classes possibles -1)
    
    output_dim_o_h : int, (default=2), est la dimension de sortie du réseau dans le cas des y 
        sous formes one_hot
    
    n_eval : float, (default=0.1), Le pourcentage de fois que le reseau calcul l'accuracy en fonction du nombre
        d'itérations. Nous pouvons aussi fournir un nombre qui signifie qu'à chaque fois qu'on 
        apprend n_eval fois alors on calcule l'accuracy
            
    path_save_best_model : String, (default=None), chemin vers le fichier où on
            veut enregister les caractéristiques du meilleur réseau avec la bibliothèque
            shelve.
    
    interval_init_weight : list(shape=2), intervalle lors de l'initialisation des poids. e.g : [-1, 1]
    
    best_accuracy : float, La meilleur accuracy qu'a obtenu une structure
        
    best_struct : dico d'éléments, La structure qui a obtenu cette meilleure accuracy
        On y retrouve ces éléments : {C, N, f, loss, learning_rate, grad_optimizer, n_iter}
    
    best_params : dico contenant des tf.Varaible, représente les poids, biais et sortie de la meilleure
        structure de réseau calculée.
    
    results : liste, contient l'ensemble des résultats en associant les paramètres d'une réseau à son accuracy.
    
    trained : bool, désigne si la structure du modèle a été entraîné au préalable. Sert pour l'utilisation
        de certaines s fonctions qui nécessites un apprentissage avant d'être utilsiée.
        
    args_optimizer : dictionnary, tous les paramètres possibles pour les différentes méthodes de descente de gradient
        initialisé à la valeur par défaut définit par la méthode de training
        
    """
    def __init__(self, output_dim=1, output_dim_o_h=2, n_eval=0.1, 
                 path_save_best_model = None):
        self.output_dim = int(output_dim)
        self.output_dim_o_h = int(output_dim_o_h)
        self.n_eval = n_eval
        self.path_save_best_model = path_save_best_model
        self.interval_init_weight = [-1.0, 1.0]
        self.best_accuracy = -1.0
        self.best_struct = {}
        self.best_params = {}
        self.results = []
        self.trained = False
        self.args_optimizer={'use_locking':False, 'initial_accumulator_value':0.1, 'beta1':0.9, 'beta2':0.999, 
            'epsilon':1e-08, 'learning_rate_power':-0.5, 'initial_accumulator_value':0.1, 'l1_regularization_strength':0.0,
            'l2_regularization_strength':0.0, 'decay':0.9, 'momentum':0.0
        }

    ###################################################################################################################################
    ###################################################################################################################################
    ######################################################## FONCTIONS UTILES #########################################################
    ###################################################################################################################################
    ###################################################################################################################################        

    """
    one_hot, crée la nouvelle variable de sortie sous forme one hot. Utile pour utiliser le loss 
    cross_entropy.
    
    Entrée :
        y : np.array ou list ou pd.Serie. Contient les différentes sorties en terme de classe,
            donc ces données sont discrères.
            
    Sortie :
        y_res : np.array. Représentant les données discrétisée sous forme one hot.
    """
    def one_hot(self):
        values = set(self.y)
        y_res = np.zeros((len(self.y), len(values)))
        for num, value in enumerate(values):
            o_h = np.zeros(len(values))
            o_h[num]=1
            y_res[self.y==value] = o_h
        return y_res
    
    """
    Renvoie un y en fonction de la fonction d'activation et du loss
    """
    def set_output(self, f, loss):
        if(f==tf.sigmoid):
            y = self.y_sig
        else:
            y = self.y_tanh
        if(loss=='cross_entropy'):
            y = self.y_o_h
        return y
    
    """
    Renvoie une dimension de sortie en fonction du loss
    """
    def set_output_dim(self, loss):
        if(loss=='cross_entropy'):
            return self.output_dim_o_h
        return self.output_dim
                           
    """
    fit_X_y permet de définir les X y de train, de validation et autres variables utiles pour l'utilisation
    de la fonction fit.
    """
    def fit_X_y(self, X, y, p_train=0.8, p_valid=None, one_hot=True):
        self.X = np.array(X)
        self.y = np.array(y)
        
        if(p_train>1.0 or p_valid>1.0):
            print "Il faut donner un pourcentage de données de train ou de validation"
            return
        
        if(p_train is None):
            p_train=1
        if(p_valid is None):
            p_valid = 1-p_train
        
        #ind_train et ind_valid vont permettre de faire le tirage aléatoire des données
        #d'entrainement et de validation.
        self.n_train = int(len(X)*p_train)
        self.n_valid = int(len(X)*p_valid)
        #Nous vérifions que nous avons bien des données de train et validation
        if(self.n_train==0):
            self.n_train==1
        if(self.n_valid==0):
            self.n_valid==1
        
        self.ind_train = np.arange(self.n_train)
        if(p_valid == 1-p_train):
            self.ind_valid = np.arange(self.n_train, self.n_train+self.n_valid)
        else:
            self.ind_valid = np.arange(self.n_valid)
        self.n_dim = self.X.shape[1]
        
        #Création des y en one_hot en one hot pour le cas cross_entropy
        if(one_hot):
            self.y_o_h = self.one_hot()
        else:
            self.y_o_h = self.y
        self.output_dim_o_h = self.y_o_h.shape[1]
        #Création des y centrés en -1 et 1 pour tanh et 0 et 1 pour sigmoide
        outputs = list(set(self.y))
        if(1 in outputs):
            self.y_tanh = np.where(self.y==1,1,-1)
            self.y_sig = np.where(self.y==1,1,0)
        if(len(outputs)<=2):
            self.y_tanh = np.where(self.y==outputs[0],1,-1)
            self.y_sig = np.where(self.y==outputs[0],1,0)
            
    """
    define_grad_optimizer, renvoie la fonction d'optimisation du gradient  en fonction du noms et des arguments
    ps : Toutes ces fonctions n'ont pas les meme paramètres.
    
    Toutes ces fonctions sont possibles:
    tf.train.MomentumOptimizer.__init__(learning_rate, momentum, use_locking=False, name='Momentum')
    tf.train.AdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad')
    tf.train.GradientDescentOptimizer.__init__(learning_rate, use_locking=False, name='GradientDescent')
    tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    tf.train.FtrlOptimizer.__init__(learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='Ftrl')
    tf.train.RMSPropOptimizer.__init__(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')
    
    Entrée:
        grad_optimizer, fonction tensorflow, la méthode d'optimisation
        learning_rate, float, le pas d'apprentissage.
    Sortie:
        La méthode grad_optimizer avec les paramètres associée.
    """
    def define_grad_optimizer(self, grad_optimizer, learning_rate):
        #GradientDescentOptimizer
        if(grad_optimizer==tf.train.GradientDescentOptimizer):
            return tf.train.GradientDescentOptimizer(learning_rate, self.args_optimizer['use_locking'])
        #AdagradOptimizer
        if(grad_optimizer==tf.train.AdagradOptimizer):
            return tf.train.AdagradOptimizer(learning_rate, self.args_optimizer['initial_accumulator_value'], self.args_optimizer['use_locking'])
        #AdamOptimizer
        if(grad_optimizer==tf.train.AdamOptimizer):
            return tf.train.AdamOptimizer(learning_rate, self.args_optimizer['beta1'], self.args_optimizer['beta2'], self.args_optimizer['epsilon'], self.args_optimizer['use_locking'])
        #MomentumOptimizer
        if(grad_optimizer==tf.train.MomentumOptimizer):
            return tf.train.MomentumOptimizer(learning_rate, self.args_optimizer['momentum'], self.args_optimizer['use_locking'])
        #FtrlOptimizer
        if(grad_optimizer==tf.train.FtrlOptimizer):
            return tf.train.FtrlOptimizer(learning_rate, self.args_optimizer['learning_rate_power'], self.args_optimizer['initial_accumulator_value'], self.args_optimizer['l1_regularization_strength'], self.args_optimizer['l2_regularization_strength'], self.args_optimizer['use_locking'])
        #RMSPropOptimizer
        if(grad_optimizer==tf.train.GradientDescentOptimizer):
            return tf.train.GradientDescentOptimizer(learning_rate, self.args_optimizer['decay'], self.args_optimizer['momentum'], self.args_optimizer['epsilon'], self.args_optimizer['use_locking'])
        print "Ce n'est pas un bon optimizer donné en paramètre"
        
    """
    set_interval_init_weight, change l'intervalle de 
    
    Entrée:
        new_interval, liste(shape=2), e.g [-1,1], [a, b] avec a<b
    """
    def set_interval_init_weight(self, new_interval):
        new_interval = map(float, new_interval)
        self.interval_init_weight = new_interval
    
    """
    set_args_optimizer, change les paramètres des méthode d'optimisation du réseau de neurones
    
    Entrée:
        **kwargs, les arguments à modifier, e.g : epsilon=1e-2, momentum=0.6...
    """
    def set_args_optimizer(self, **kwargs):
        for args in kwargs:
            self.args_optimizer[args]=kwargs[args]
    
    """
    early_stopping, stop l'apprentissage du réseau de neurone s'il n'apprend plus

    Entrée:
        p: int, le nombre de patience de l'algorithme
    Sortie
        should_stop: boolean, si Vrai alors l'apprentissage s'arrête sinon il continue.
    """
    def early_stopping(self, p, accuracy, n_eval, teta, i, j, v, teta_star, i_star):
        should_stop=False
        i=i+n_eval
        v_prim = accuracy
        if(v_prim < v):
            j=0
            teta_star=teta
            i_star=i
            v=v_prim
        else:
            j=j+1
        if(j>=p):
            should_stop = True
            return should_stop, teta, i, j, v, teta_star, i_star
        return should_stop, teta, i, j, v, teta_star, i_star
            
    """
    create_NN, Fonction qu permet de créer un dictionnaire contenant tout les poids 
    les biais et les sorties de notre réseau de neurones (de la couche d'entrée
    à la couche de sortie). Pour l'instant le réseau créé est de type MLP.
    
    Entréess :
        X : placeholder, représente les données passées au réseau
        
        C : int, Nombre de couches de réseau
        
        N : tab de int, Nombre de neurones par couche, le premier chiffre
            correspond à la couche d'entrée tandis que le dernier chiffre
            correspond à la couche de sortie (entre ce sont les couches
            cachées)
        
        f : fonction de type TensorFlow, la fonction d'activation du réseau
            qui sera utilisée. En général tf.sigmoid, tf.tanh etc...
            
        loss : String, correspond à la fonction de cout à optimiser pour 
            entraîner le réseau.
            possibilités : "mse", "hinge" ou "cross_entropy"
            "mse" et "hinge" pour la classif binaire et "corss_entropy" pour 
            binaire ou multi classe, sachant que les sortie doivent etre de la
            forme d'une liste (ou tableau) one hot.
        
        n_dim : int, est la dimension des données à apprendre (utile pour
            la couche d'entrée)
            
        output_dim : int, est la dimension de sortie du réseau (correspond au
            nombre de classes possibles -1)
    
    Sorties : 
        dico : dictionnaire, contient des tf.Variable représentants les poids,
            biais et sorties des neurones. C'est variables sont numérotés de la 
            sortes : wij avec i représentant la couche du réseau (0 pour la 
            première couche) et j représentant le neurone j de la couche i.
    """

    def create_NN(self, X, C, N, f, loss, output_dim):
        #Dictionnaire de sortie contenant les variables
        dico = {}
        
        #Contient la liste des output du réseau entier regroupés par couche,
        #exemple : [[y00, y01, y02],[y10,y11]]
        list_y_NN = []
        # Arrête la construction du reseau si une couche n'a pas de neurones
        stop_build = False
        
        try:
            N = N + [1]
        except:
            N = [1]
        C = len(N)
        proj = 1
        for c in range(C):
            #liste des output pour une couche.
            list_y = []
            #Nous parcourons les neurones sur la couche c 
            for n in range(N[c]):
                #Couche de entrée et sortie
                if(c==C-1 and c==0):
                    # Définition des poids, biais et sortie
                    dico["w"+str(c)+str(n)] = tf.Variable(tf.random_uniform([self.n_dim, output_dim], self.interval_init_weight[0], self.interval_init_weight[1]))
                    dico["b"+str(c)+str(n)] = tf.Variable(tf.zeros([output_dim]))
                    if(loss=='mse' or loss=='hinge'):
                        with tf.name_scope("prediction") as scope:
                            if(f!=tf.sigmoid and f!=tf.tanh):
                                f=tf.sigmoid
                            dico["y"+str(c)+str(n)] = f(tf.matmul(X, dico["w"+str(c)+str(n)]) + dico["b"+str(c)+str(n)])
                    else:
                        with tf.name_scope("prediction") as scope:
                            dico["y"+str(c)+str(n)] = tf.nn.softmax(tf.matmul(X, dico["w"+str(c)+str(n)]) + dico["b"+str(c)+str(n)])
                    # Ajout de y à la list pour la couche actuelle
                    list_y.append(dico["y"+str(c)+str(n)])
                                        
                #Couche de sortie
                elif(c==C-1):
                    # Définition des poids, biais et sortie
                    dico["w"+str(c)+str(n)] = tf.Variable(tf.random_uniform([N[c-1]*proj, output_dim], self.interval_init_weight[0], self.interval_init_weight[1]))
                    dico["b"+str(c)+str(n)] = tf.Variable(tf.zeros([output_dim]))
                    if(loss=='mse' or loss=='hinge'):
                        with tf.name_scope("prediction") as scope:
                            if(f!=tf.sigmoid and f!=tf.tanh):
                                f=tf.sigmoid
                            dico["y"+str(c)+str(n)] = f(tf.matmul(tf.concat(1, list_y_NN[c-1]), dico["w"+str(c)+str(n)]) + dico["b"+str(c)+str(n)])
                    else: #Cas cross_entropy
                        with tf.name_scope("prediction") as scope:
                            dico["y"+str(c)+str(n)] = tf.nn.softmax(tf.matmul(tf.concat(1, list_y_NN[c-1]), dico["w"+str(c)+str(n)]) + dico["b"+str(c)+str(n)])
                    # Ajout de y à la list pour la couche actuelle
                    list_y.append(dico["y"+str(c)+str(n)])
                    
                #Couche d'entrée
                elif(c==0):
                    # Défintion des poids
                    dico["w"+str(c)+str(n)] = tf.Variable(tf.random_uniform([self.n_dim, proj], self.interval_init_weight[0], self.interval_init_weight[1]))
                    # Définition du biais
                    dico["b"+str(c)+str(n)] = tf.Variable(tf.zeros([proj]))
                    # Calcul de la sortie y
                    with tf.name_scope("couche_entree") as scope:
                        dico["y"+str(c)+str(n)] = f(tf.matmul(X, dico["w"+str(c)+str(n)]) + dico["b"+str(c)+str(n)])
                    # Ajout de y à la list pour la couche actuelle
                    list_y.append(dico["y"+str(c)+str(n)])
                    
                #Couche(s) cachée(s)
                else:
                    # Définition des poids
                    dico["w"+str(c)+str(n)] = tf.Variable(tf.random_uniform([N[c-1]*proj, proj], self.interval_init_weight[0], self.interval_init_weight[1]))
                    # Définiton du biais
                    dico["b"+str(c)+str(n)] = tf.Variable(tf.zeros([proj]))
                    # Calcul de la sortie y
                    with tf.name_scope("couches_cachees") as scope:
                        dico["y"+str(c)+str(n)] = f(tf.matmul(tf.concat(1, list_y_NN[c-1]), dico["w"+str(c)+str(n)]) + dico["b"+str(c)+str(n)])
                    # Ajout de y à la list pour la couche actuelle
                    list_y.append(dico["y"+str(c)+str(n)])
            list_y_NN.append(list_y)
        return dico

    ###################################################################################################################################
    ###################################################################################################################################
    ########################################### PARTIE APPRENTISSAGE DU RESEAU DE NEURONES ############################################
    ###################################################################################################################################
    ###################################################################################################################################

                    
    """
    fit, Fonction qui va entraîner le réseau de neurones avec les caractéristiques 
    passées en paramètre. Met a jour les variables best_accuracy et best_struct 
    si le résultat obtenu est meilleur.
    
    Entrées :
        X : array, default=None, Données d'apprentissage. Si non alors on utilise le X et le y de la variable de classe.
        y : array, default=None, labels d'apprentissage. Si non alors on utilise le X et le y de la variable de classe.
        verif : boolean, default=False, Si on veut vérifier la descente de gradient (toujours à vrai lors de l'entrainement de la structure) 
        p_train : default = 1, utile si verif est True pourcentage. Lorsque cette variable vaut 1, toutes les données sont utilisées en train.
        p_valid=None : utile si verif est True. Détermine le pourventage de données utilisées en validation
        
        C : int, default=1, Nombre de couches de réseau
        
        N : tab/list de int, default = 1, Nombre de neurones par couche, le premier chiffre
            correspond à la couche d'entrée tandis que le dernier chiffre
            correspond à la couche de sortie (entre ce sont les couches
            cachées)
            
        f : fonction de type TensorFlow, default=tf.sigmoid. Est la fonction
            d'activation du réseau qui sera utilisée.
            En général tf.sigmoid, tf.tanh etc...
            
        loss : String, default='mse', correspond à la fonction de cout à optimiser 
            pour entraîner le réseau.
            possibilités : "mse", "hinge" ou "cross_entropy"
            "mse" et "hinge" pour la classif binaire et "corss_entropy" pour 
            binaire ou multi classe, sachant que les sortie doivent etre de la
            forme d'une liste (ou tableau) one hot.
            
        learning_rate : float, default=1e-3. Est le pas d'apprentissage du réseau.
        
        n_iter : int, default=1000, est le nombre d'itération de l'apprentissage du réseau.
        
        batch_train : int, default=1 (méthode stochastique) .Nombre de 
            données passées à l'étape de train.
            
        batch_valid : int, default=np.Inf (toute les données de validation 
            sont fournies). Nombre de données passées au moment du calcul de 
            l'accuracy.
            
        shuffle : boolean, default=False. Si vrai, utilise un fichier 
            bash qui mélange les données d'un fichier au format csv, si non
            ne fait rien.
        
        early_stopping : boolean, deafault=False, détermine si on arrête prépaturément l'apprentissage du réseau
        
        p_early_stopping : int, default=3, nombre de patience de l'algorithme early_stopping, si early_stopping=False,
            alors ce paramètre est inutile.
            
        n_threads : int, default=1. Nombre de thread que l'on veut utiliser pour
            nos opérations. Si pas supérieur à 0 alors on ne change rien.
            
        num_gpu : int, default=0, le numéro du gpu à utiliser. Le gpu 0 est utilisé par défaut,
            cependant si la machine n'a pas de gpu, l'option allow_soft_placement permet d'éviter
            la collision de device.

        display : boolean, dafault=False, Si vrai, affiche l'accuracy à chaque fois
            qu'elle est calculée. Si non ne fait rien.
        
    Sortie : None
    """

    def fit(self, X=None, y=None, verif=False, p_train=None, p_valid=None, C=1, N=[1], f=tf.sigmoid, loss='mse', grad_optimizer=tf.train.GradientDescentOptimizer, learning_rate=1e-3, n_iter=1000,
            batch_train=1, batch_valid=np.Inf, early_stopping=False, p_early_stopping=3,
            n_threads=1, num_gpu=0, display=False):
        
        #Nous testons si le but de l'exécution est de faire un affichage de la descente de gradient
        if(display==True):
            verif=True
        if(p_train is None):
            p_train=0.8
            p_valid=None
        
        #Nous testons si X et y sont None, si ce n'est pas le cas on fait appelle à fit_X_y pour changer les variables de la classe
        if(not X is None and not y is None):
            self.fit_X_y(X, y, p_train=p_train, p_valid=p_valid, one_hot=True)
        
        #Nous déterminons quelles sorties nous allons utiliser (cas tanh, sigmoid ou cross entropy)
        y = self.set_output(f, loss)
        output_dim = self.set_output_dim(loss)
        
        #Configuration de la session du graphe de tensorflow
        config = tf.ConfigProto()
        #config.gpu_options.allocator_type = 'BFC'
        config.allow_soft_placement=True
        config.intra_op_parallelism_threads=n_threads
        config.inter_op_parallelism_threads=n_threads
        #Définition de la session
        with tf.Session(config=config) as sess:
            #Définition du gpu à utiliser
            with tf.device('/gpu:'+str(num_gpu)):
                #Défintion de l'entrée du réseau 
                X = tf.placeholder(tf.float32, shape=[None, self.n_dim], name="X_input")
                y_ = tf.placeholder(tf.float32, shape=[None, output_dim], name="y_input")
                #Création du dictionnaire représentant le réseau de neurones
                NN = self.create_NN(X, C, N, f, loss, output_dim)

                #Formule du coût à optimiser
                if(loss=='mse'):
                    l=tf.reduce_mean(tf.square(NN["y"+str(C-1)+str(N[C-1]-1)]-y_))
                elif(loss=='hinge'):
                    l=tf.reduce_mean(tf.maximum(0.0, 1-NN["y"+str(C-1)+str(N[C-1]-1)]*y_))
                else: #cas cross entropy
                    l=-tf.reduce_sum(y_*tf.log(NN["y"+str(C-1)+str(N[C-1]-1)]))
    
                #Optimisation par descente de gradient
                train_step = self.define_grad_optimizer(grad_optimizer, learning_rate).minimize(l)
                
                #Calcul de la prédiction et de l'accuracy
                if(f==tf.tanh and (loss=='mse' or loss=='hinge')):
                    correct_prediction = tf.equal(tf.sign(NN["y"+str(C-1)+str(N[C-1]-1)]), y_)
                elif(f==tf.sigmoid and (loss=='mse' or loss=='hinge')):
                    correct_prediction = tf.equal(tf.cast(NN["y"+str(C-1)+str(N[C-1]-1)]>0.5, tf.float32), y_) #Recentrage entre 0 et 1 pour le cas sigmoid
                else:
                    correct_prediction = tf.equal(tf.argmax(NN["y"+str(C-1)+str(N[C-1]-1)],1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
                #Calcul du nombre d'évaluation à effectuer lors de l'apprentissage
                if(self.n_eval<=1 and self.n_eval>=0):
                    n_eval = int(n_iter/(self.n_eval*100))
                else:
                    n_eval = int(np.abs(self.n_eval))
                    
                #On regarde si les batch ne pas trop grand
                if(batch_train>self.n_train):
                    batch_train = self.n_train
                if(batch_valid>self.n_valid):
                    batch_valid = self.n_valid
                
                #itération, old_loss et new_loss pour déterminer l'arrête de l'appentissage ou non
                iteration=0
                acc =-1
                
                #Utile pour le early stopping
                if(early_stopping):
                    i=0
                    j=0
                    v=np.Inf
                    teta_star = NN
                    i_star=i
                
                #Initialisation de toutes nos variable contenu dans le dictionnaire
                sess.run(tf.initialize_all_variables())
                
                #On append jusqu'à convergence ou un nombre maximum d'itération
                should_stop = False
                while(iteration<=n_iter and not should_stop):
                    #Calcul de l'accuracy
                    if(iteration % n_eval == 0 and verif):
                        #On choisit aléatoirement un certain nombre de données de validation
                        xi = np.random.choice(self.ind_valid, batch_valid, replace=False)
                        #Nous remplissons les placeholders
                        feed = {X: self.X[xi].reshape((batch_valid,self.X.shape[1])), y_: y[xi].reshape((batch_valid, output_dim))}
                        #On calcul a la fois l'accuracy et le loss
                        #l'accuracy permet de voir si le modèle est performant
                        #le loss est un contrôle pour voir si la descente de gradient fonctionne correctement
                        result = sess.run([accuracy, l], feed_dict=feed)
                        acc = result[0]
                        loss_res = result[1]
                        #Nous déterminons si nous devons stopper l'apprentissage
                        if(early_stopping):
                            should_stop, teta, i, j, v, teta_star, i_star = self.early_stopping(p_early_stopping, acc, n_eval, NN, i, j, v, teta_star, i_star)
                        #On affiche ou pas le résultat de l'accuracy.
                        if(display):
                            print("At step %s, loss=%s, accuracy=%s" %(iteration, loss_res, acc))
                    #Aprentissage du modèle
                    else:
                        #On choisit aléatoirement un certain nombre de données de validation (en général 1 pour la méthode stochastic)
                        xi = np.random.choice(self.ind_train, batch_train, replace=False)
                        #Nous remplissons les placeholders
                        feed = {X: self.X[xi].reshape((batch_train,self.X.shape[1])), y_: y[xi].reshape((batch_train, output_dim))}
                        #Enfin on apprend le réseau et on met les poids à jour
                        sess.run(train_step, feed_dict=feed)
                    iteration+=1
                #On enregistre le résultat du NN
                self.results.append([acc, {'C':C, 'N':N, 'f':f, 'loss':loss, 'learning_rate':learning_rate, 'grad_optimizer': grad_optimizer, 'n_iter':n_iter}])
                #On met à jour la best_accuracy, la best_struct et le best_params de la classe si ils ont été battus 
                if(self.best_accuracy < acc):
                    self.best_accuracy = acc
                    self.best_struct = {'C':C, 'N':N, 'f':f, 'loss':loss, 'learning_rate':learning_rate, 'grad_optimizer': grad_optimizer, 'n_iter':n_iter}
                    self.best_params = {}
                    for key, value in NN.iteritems():
                        if(key[0]=='y'):continue
                        self.best_params[key] = sess.run(value)
                    #RAJOUTER LA SAUVEGARDE DU DICTIONNAIRE
                    # Si on doit enregistrer le modèle alors on le fait à l'endroit spécifié
                    if(self.path_save_best_model is not None):
                        f_saver = shelve.open(self.path_save_best_model)
                        f_saver['best_accuracy'] = self.best_accuracy
                        f_saver['best_struct'] = self.best_struct
                        f_saver['best_params'] = self.best_params
                        f_saver.close()  


    """
    fit_best, sur couche de fit où cette fois la classe va créer un réseau avec les paramètres contenus dans best_models.
        Ne peut être utilisée que si self.trained est True.
    """
    def fit_best(self, X=None, y=None, verif=False, p_train=None, p_valid=None, batch_train=1, batch_valid=np.Inf, early_stopping=False, p_early_stopping=3, n_threads=1, num_gpu=0, display=False):
        if(self.trained==False):
            print "La structure du réseau n'a pas était apprise, apprenez la avant."
            return
        
        C = self.best_struct['C']
        N = self.best_struct['N']
        f = self.best_struct['f']
        loss = self.best_struct['loss']
        learning_rate = self.best_struct['learning_rate']
        grad_optimizer=self.best_struct['grad_optimizer']
        n_iter = self.best_struct['n_iter']
        
        self.fit(X=X, y=y, verif=verif, p_train=p_train, p_valid=p_valid, 
                 C=C, N=N, f=f, loss=loss, grad_optimizer=grad_optimizer, learning_rate=learning_rate, n_iter=n_iter,
            batch_train=batch_train, batch_valid=batch_valid,early_stopping=early_stopping, 
            n_threads=n_threads, num_gpu=num_gpu, display=display)

    """
    fit_plot : Equivalent de fit mais avec le paramètre path_save_summarie correspondant au dossier où l'on veut
        enregistrer les summaries dans le dossier spécifié. Si on préfère passeer par une méthode avec plot via 
        matplot lib il faut juste laisser path_save_summeries à None.
    """
    def fit_plot(self, X=None, y=None, p_train=None, p_valid=None, path_save_summaries=None, C=1, N=[1], f=tf.sigmoid, loss='mse', grad_optimizer=tf.train.GradientDescentOptimizer, learning_rate=1e-3, n_iter=1000,
            batch_train=1, batch_valid=np.Inf, early_stopping=False, p_early_stopping=3, n_threads=1, num_gpu=0, display=False):
        
        #Nous testons si le but de l'exécution est de faire un affichage de la descente de gradient
        if(display==True):
            verif=True
        if(p_train is None):
            p_train=0.8
            p_valid=None
        
        #Nous testons si X et y sont None, si ce n'est pas le cas on fait appelle à fit_X_y pour changer les variables de la classe
        if(not X is None and not y is None):
            self.fit_X_y(X, y, p_train=p_train, p_valid=p_valid, one_hot=True)
        
        #Nous déterminons quelles sorties nous allons utiliser (cas tanh, sigmoid ou cross entropy)
        y = self.set_output(f, loss)
        output_dim = self.set_output_dim(loss)
        
        #Configuration de la session du graphe de tensorflow
        config = tf.ConfigProto()
        #config.gpu_options.allocator_type = 'BFC'
        config.allow_soft_placement=True
        config.intra_op_parallelism_threads=n_threads
        config.inter_op_parallelism_threads=n_threads
        #Définition de la session
        
        with tf.Session(config=config) as sess:
            #Définition du gpu à utiliser
            with tf.device('/gpu:'+str(num_gpu)):
                #Création de la liste de résultat à plot
                self.ordonnees_acc = []
                self.ordonnees_loss = []
                self.abscisses_iter = []
                #Défintion de l'entrée du réseau 
                X = tf.placeholder(tf.float32, shape=[None, self.n_dim], name="X_input")
                y_ = tf.placeholder(tf.float32, shape=[None, output_dim], name="y_input")
                
                #Création du dictionnaire représentant le réseau de neurones
                NN = self.create_NN(X, C, N, f, loss, output_dim)

                #Formule du coût à optimiser
                if(loss=='mse'):
                    with tf.name_scope("mse_criterion") as scope:
                        l=tf.reduce_mean(tf.square(NN["y"+str(C-1)+str(N[C-1]-1)]-y_))
                        l_summ = tf.scalar_summary("mse",l)
                elif(loss=='hinge'):
                    with tf.name_scope("hinge_criterion") as scope:
                        l=tf.reduce_mean(tf.maximum(0.0, 1-NN["y"+str(C-1)+str(N[C-1]-1)]*y_))
                        l_summ = tf.scalar_summary("hinge_loss",l)
                else: #cas cross entropy
                    with tf.name_scope("cross_entropy_criterion") as scope:
                        l=-tf.reduce_sum(y_*tf.log(NN["y"+str(C-1)+str(N[C-1]-1)]))
                        l_summ = tf.scalar_summary("cross_entropy",l)

                #Optimisation par descente de gradient
                with tf.name_scope("train") as scope:
                    train_step = self.define_grad_optimizer(grad_optimizer, learning_rate).minimize(l)

                #Calcul de la prédiction et de l'accuracy
                with tf.name_scope("test") as scope:
                    if(f==tf.tanh and (loss=='mse' or loss=='hinge')):
                        correct_prediction = tf.equal(tf.sign(NN["y"+str(C-1)+str(N[C-1]-1)]), y_)
                    elif(f==tf.sigmoid and (loss=='mse' or loss=='hinge')):
                        correct_prediction = tf.equal(tf.cast(NN["y"+str(C-1)+str(N[C-1]-1)]>0.5, tf.float32), y_) #Recentrage entre -1 et 1 pour le cas sigmoid
                    else:
                        correct_prediction = tf.equal(tf.argmax(NN["y"+str(C-1)+str(N[C-1]-1)],1), tf.argmax(y_, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    
                merged = tf.merge_all_summaries()

                if(path_save_summaries!=None):
                    numFolder = 0
                    while(os.path.exists(path_save_summaries+"training_NN_"+str(numFolder))):
                        numFolder+=1
                    # Nous créons le writer des summaries avec cette fonction
                    writer = tf.train.SummaryWriter(path_save_summaries+"training_NN_"+str(numFolder), sess.graph_def)
                
                #Calcule du nombre d'évaluation à effectuer lors de l'apprentissage
                if(self.n_eval<=1 and self.n_eval>=0):
                    n_eval = self.n_eval*n_iter
                else:
                    n_eval = int(np.abs(self.n_eval))
                    
                #Calcul du nombre d'évaluation à effectuer lors de l'apprentissage
                if(self.n_eval<=1 and self.n_eval>=0):
                    n_eval = int(n_iter/(self.n_eval*100))
                else:
                    n_eval = int(np.abs(self.n_eval))
                    
                #On regarde si les batch ne pas trop grand
                if(batch_train>self.n_train):
                    batch_train = self.n_train
                if(batch_valid>self.n_valid):
                    batch_valid = self.n_valid
                
                #itération, old_loss et new_loss pour déterminer l'arrête de l'appentissage ou non
                iteration=0
                acc =-1
                
                #Utile pour le early stopping
                if(early_stopping):
                    i=0
                    j=0
                    v=np.Inf
                    teta_star = NN
                    i_star=i
                
                #Initialisation de toutes nos variable contenu dans le dictionnaire
                sess.run(tf.initialize_all_variables())
                
                #On append jusqu'à convergence ou un nombre maximum d'itération
                should_stop = False
                while(iteration<=n_iter and not should_stop):
                    #Calcul de l'accuracy
                    if(iteration % n_eval == 0):
                        #On choisit aléatoirement un certain nombre de données de validation
                        xi = np.random.choice(self.ind_valid, batch_valid, replace=False)
                        #Nous remplissons les placeholders
                        feed = {X: self.X[xi].reshape((batch_valid,self.X.shape[1])), y_: y[xi].reshape((batch_valid, output_dim))}
                        #On calcul a la fois l'accuracy et le loss
                        #le loss permet d'avoir notre critère d'arrête
                        #l'accuracy permet de voir si le modèe est performant
                        result = sess.run([accuracy, l], feed_dict=feed)
                        #On teste si on plot sur tensorboard ou matplot lib
                        if(path_save_summaries != None):
                            summary_str = sess.run(merged, feed_dict=feed)
                            writer.add_summary(summary_str, iteration)
                        #On enregistre le score et le loss
                        self.ordonnees_acc.append(result[0])
                        self.ordonnees_loss.append(result[1])
                        self.abscisses_iter.append(iteration)
                        acc = result[0]
                        loss_res = result[1]
                        #Nous déterminons si nous devons stopper l'apprentissage
                        if(early_stopping):
                            should_stop, teta, i, j, v, teta_star, i_star = self.early_stopping(p_early_stopping, acc, n_eval, NN, i, j, v, teta_star, i_star)
                        #On affiche ou pas le résultat de l'accuracy.
                        if(display):
                            print("At step %s, loss=%s, accuracy=%s" %(iteration, loss_res, acc))
                    #Aprentissage du modèle
                    else:
                        #On choisit aléatoirement un certain nombre de données de validation (en général 1 pour la méthode stochastic)
                        xi = np.random.choice(self.ind_train, batch_train, replace=False)
                        #Nous remplissons les placeholders
                        feed = {X: self.X[xi].reshape((batch_train,self.X.shape[1])), y_: y[xi].reshape((batch_train, output_dim))}
                        #Enfin on apprend le réseau et on met les poids à jour
                        sess.run(train_step, feed_dict=feed)
                    iteration+=1
                
                #On enregistre le résultat du NN
                self.results.append([acc, {'C':C, 'N':N, 'f':f, 'loss':loss, 'learning_rate':learning_rate, 'grad_optimizer': grad_optimizer, 'n_iter':n_iter}])
                #On met à jour la best_accuracy et le best_modèle de la classe si ils ont été battus 
                if(self.best_accuracy < acc):
                    self.best_accuracy = acc
                    self.best_struct = {'C':C, 'N':N, 'f':f, 'loss':loss, 'learning_rate':learning_rate, 'grad_optimizer': grad_optimizer, 'n_iter':n_iter}
                    self.best_params = {}
                    for key, value in NN.iteritems():
                        if(key[0]=='y'):continue
                        self.best_params[key] = sess.run(value)
                    # Si on doit enregistrer le modèle alors on le fait à l'endroit spécifié
                    if(self.path_save_best_model is not None):
                        f_saver = shelve.open(self.path_save_best_model)
                        f_saver['best_accuracy'] = self.best_accuracy
                        f_saver['best_struct'] = self.best_struct
                        f_saver['best_params'] = self.best_params
                        f_saver.close()
                        
                #On affiche les plot si jamais on ne veut pas les summaries        
                if(path_save_summaries == None):
                    plt.figure(1)
                    plt.title("Accuracy")
                    plt.xlabel("iterations")
                    plt.ylabel("accuracy")
                    plt.plot(self.abscisses_iter, self.ordonnees_acc)
                    plt.figure(2)
                    plt.title("Loss")
                    plt.xlabel("iterations")
                    plt.ylabel("loss")
                    plt.plot(self.abscisses_iter, self.ordonnees_loss)
                    plt.show()
    
    """
    predict, renvoie la sortie calculé par le réseau de neurones ayant eu le meilleur score.
    
    Entrées :
        X : Les données à prédire
        
    Sortie :
        predictions : Array, tableau contenant les labels prédits par le réseau
    """
    def predict(self, X):
        if(self.trained==False):
            print "La structure du réseau n'a pas était apprise, apprenez la avant."
            return
        X = np.array(X)
        #Placeholder auquel nous passerons les X
        X_ = tf.placeholder(tf.float32, shape=[None, self.n_dim], name="X_input")
        #Nous sélectionnons les paramètres utiles pour la prédiction
        C = self.best_struct['C']
        N = self.best_struct['N']
        f = self.best_struct['f']
        loss = self.best_struct['loss']
        #Récupération des paramètres du modèle
        NN = self.best_params
        with tf.Session() as sess:
            list_y_NN = []
            for c in range(C):
                list_y=[]
                for n in range(N[c]):
                    if(c==C-1 and c==0):
                        if(loss=='mse' or loss=='hinge'):
                            output = f(sigmoid(tf.matmul(X_, NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)]))
                        else: #cas cross_entropy
                            output = tf.nn.softmax(tf.matmul(X_, NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)])
                    elif(c==C-1):
                        if(loss=='mse' or loss=='hinge'):
                            output = f(tf.matmul(tf.concat(1, list_y_NN[c-1]), NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)])
                        else:
                            output = tf.nn.softmax(f(tf.matmul(tf.concat(1, list_y_NN[c-1]), NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)]))
                    elif(c==0):
                        y = f(tf.matmul(X_, NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)])
                        list_y.append(y)
                    else:
                        y = f(tf.matmul(tf.concat(1, list_y_NN[c-1]), NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)])
                        list_y.append(y)
                list_y_NN.append(list_y)

            if(f==tf.tanh and (loss=='mse' or loss=='hinge')):
                prediction = tf.sign(output)
            elif(f==tf.sigmoid and (loss=='mse' or loss=='hinge')):
                prediction = tf.cast(output>0.5, tf.float32)
            else:
                prediction = tf.argmax(output,1)
            if(len(X.shape)==1):
                feed = {X_: X.reshape((1,X.shape[0]))}
            else:
                feed = {X_: X}
            predictions = np.ravel(sess.run(prediction, feed_dict=feed))
        return predictions
    
    """
    Idem que prédict mais renvoie une probabilité au lieu de la valeur du label.
    ps : Ne marche que pour cross entropy
    """
    def predict_proba(self, X):
        if(self.trained==False):
            print "La structure du réseau n'a pas était apprise, apprenez la avant."
            return
        X = np.array(X)
        #Placeholder auquel nous passerons les X
        X_ = tf.placeholder(tf.float32, shape=[None, self.n_dim], name="X_input")
        #Nous sélectionnons les paramètres utiles pour la prédiction
        C = self.best_struct['C']
        N = self.best_struct['N']
        f = self.best_struct['f']
        loss = self.best_struct['loss']
        if(not loss=="cross_entropy"):
            print "Seul avec le loss cross entropy il est possible de renvoyer une probabilité"
            return
        #Récupération des paramètres du modèle
        NN = self.best_params
        with tf.Session() as sess:
            list_y_NN = []
            for c in range(C):
                list_y=[]
                for n in range(N[c]):
                    if(c==C-1 and c==0):
                        output = tf.nn.softmax(tf.matmul(X_, NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)])
                    elif(c==C-1):
                        output = tf.nn.softmax(f(tf.matmul(tf.concat(1, list_y_NN[c-1]), NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)]))
                    elif(c==0):
                        y = f(tf.matmul(X_, NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)])
                        list_y.append(y)
                    else:
                        y = f(tf.matmul(tf.concat(1, list_y_NN[c-1]), NN["w"+str(c)+str(n)]) + NN["b"+str(c)+str(n)])
                        list_y.append(y)
                list_y_NN.append(list_y)

            prediction = output
            if(len(X.shape)==1):
                feed = {X_: X.reshape((1,X.shape[0]))}
            else:
                feed = {X_: X}
            predictions = sess.run(prediction, feed_dict=feed)
        return predictions
    
    """
    score, moyenne le nombre de bonnes prédictions de la fonction predict
    
    Entrées :
        X : Les données à prédire
        y : les vraies valeurs des données
        
    Sortie :
        predictions : float, pourcentage représentant le score
    """
    def score(self, X, y):
        predictions = self.predict(X)
        return (y==predictions).mean()
                    
    ###################################################################################################################################
    ###################################################################################################################################
    ############################################# PARTIE APPRENTISSAGE STRUCTURE DU RESEAU ############################################
    ###################################################################################################################################
    ###################################################################################################################################

    
    """
    rnn_nn : Cas non multi_thread, fit n_NN fois le grid_search
    """

    def run_NN(self, n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, n_iter_list,
                      batch_train, batch_valid, early_stopping, p_early_stopping, n_threads, num_gpus, p_show):
        #liste permettant de tester si une structure a été entraîné
        N_done=[]
        #On inverse la liste pour facilité le parcours
        N_list.reverse()
        #Compteur de réseau
        NN_cpt=0     
        #Récupération du nombre de couches
        while(NN_cpt!=n_NN):
            for C in C_list:
                N = [] #Le reseau qu'on crée
                for num_layer in range(C):
                    N.insert(0, np.random.choice(N_list[num_layer]))
                #On vérifie que le réseau créé aléatoirement n'a pas déjà été testé
                if(N not in N_done):
                    N_done.append(N)
                else:
                    NN_cpt+=1
                    continue
                for f in f_list:
                    for loss in loss_list:
                        for grad_optimizer in grad_optimizer_list:
                            for learning_rate in learning_rate_list:
                                for n_iter in n_iter_list:
                                    #On sélectionne le gpu aléatoirement s'il y en a plusieurs
                                    if(isinstance(num_gpus,list)):
                                        num_gpu = np.random.choice(num_gpus)
                                    else:
                                        num_gpu = num_gpus
                                    #On ne peut pas utiliser le loss hinge avec la fonction sigmoid car elle
                                    #est centrée en 0 et 1
                                    if(f==tf.sigmoid and loss=="hinge"):
                                        f=tf.tanh
                                    #Nous devons utiliser la sigmoid dans le cas d'un coup cross entropy
                                    if(f!=tf.sigmoid and loss=='cross_entropy'):
                                        f=tf.sigmoid
                                    #On lance l'apprentissage du réseau
                                    self.fit(None, None, True, None, None, C, N, f, loss, grad_optimizer, learning_rate, n_iter, batch_train, batch_valid, 
                                             early_stopping, p_early_stopping, n_threads, num_gpu, False) #display=False dans le grid search
                                    
                                    NN_cpt+=1
                                    if(NN_cpt%(int(p_show*n_NN)+1)==0):
                                        print NN_cpt
                                    if(NN_cpt >= n_NN): 
                                        return
    
    """
    create_threads : Cas multi_thread, se contente de créer la liste de thread et de la retourner
    """
    def create_threads(self, n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, n_iter_list,
                      batch_train, batch_valid, early_stopping, p_early_stopping, n_threads, num_gpus):
        #Création de la liste des processus
        threads=[]
        #liste permettant de tester si une structure a été entraîné
        N_done=[]
        #On inverse la liste pour facilité le parcours
        N_list.reverse()
        #Compteur de réseau
        NN_cpt=0
        #Récupération du nombre de couches
        while(NN_cpt != n_NN):
            for C in C_list:
                N = [] #Le reseau qu'on crée
                for num_layer in range(C):
                    N.insert(0, np.random.choice(N_list[num_layer]))
                #On vérifie que le réseau créé aléatoirement n'a pas déjà été testé
                if(N not in N_done):
                    N_done.append(N)
                else:
                    NN_cpt+=1
                    continue
                for f in f_list:
                    for loss in loss_list:
                        for grad_optimizer in grad_optimizer_list:
                            for learning_rate in learning_rate_list:
                                for n_iter in n_iter_list:
                                    #On sélectionne le gpu aléatoirement s'il y en a plusieurs
                                    if(isinstance(num_gpus,list)):
                                        num_gpu = np.random.choice(num_gpus)
                                    else:
                                        num_gpu = num_gpus
                                    #On ne peut pas utiliser le loss hinge avec la fonction sigmoid car elle
                                    #est centrée en 0 et 1
                                    if(f==tf.sigmoid and loss=="hinge"):
                                        f=tf.tanh
                                    #Nous devons utiliser la sigmoid dans le cas d'un coup cross entropy
                                    if(f!=tf.sigmoid and loss=='cross_entropy'):
                                        f=tf.sigmoid
                                    #Création de la liste de paramètre pour la donenr au thread
                                    param_thread = [None, None, True, None, None, C, N, f, loss, grad_optimizer, learning_rate, int(n_iter),
                                                    int(batch_train), int(batch_valid), early_stopping,
                                                    p_early_stopping, int(n_threads), num_gpu, False] #display=False dans le grid search

                                    #Ajout du thread à la liste
                                    threads.append(tf.train.threading.Thread(target=self.fit, args=param_thread))
                                    #Test d'arrête si nous avons construit suffisamment de réseaux
                                    NN_cpt+=1
                                    if(NN_cpt >= n_NN):
                                        return threads  
        return threads
    
    """
    run_NN_multi_local, cas non multi-thread, lance l'apprentissage des NN en distribué localement
    """
    def run_NN_multi_local(self, n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, 
                                n_iter_list, batch_train, batch_valid, early_stopping, p_early_stopping,
                                n_threads, num_gpus):
        #Création du coordinateur qui va coordonner les thread
        coord = tf.train.Coordinator()
        #Création de la liste des processus
        threads=self.create_threads(n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, 
                                n_iter_list, batch_train, batch_valid, early_stopping, p_early_stopping,
                                n_threads, num_gpus)
            
        cpt_threads=0
        while(cpt_threads < len(threads)):
            cpt=0
            while(cpt<n_threads and cpt_threads<len(threads)):
                threads[cpt_threads].start()
                cpt_threads+=1
                cpt+=1
            coord.join(threads)
            
    ###################################################################################################################################
    ###################################################################################################################################
    ######################################## PARTIE APPRENTISSAGE STRUCTURE DU RESEAU AVEC SPARK ######################################
    ###################################################################################################################################
    ###################################################################################################################################
         
    """
    create_list_params: crée une liste de paramètres qui sera parallelisé par le sparkcontext
    """
    def create_list_params(self, n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, 
                                n_iter_list, batch_train, batch_valid, early_stopping, p_early_stopping,
                                n_threads, num_gpus):
        #Création de la liste des processus
        list_params=[]
        #liste permettant de tester si une structure a été entraîné
        N_done=[]
        #On inverse la liste pour facilité le parcours
        N_list.reverse()
        #Compteur de réseau
        NN_cpt=0
        #Récupération du nombre de couches
        while(NN_cpt != n_NN):
            for C in C_list:
                N = [] #Le reseau qu'on crée
                for num_layer in range(C):
                    N.insert(0, np.random.choice(N_list[num_layer]))
                #On vérifie que le réseau créé aléatoirement n'a pas déjà été testé
                if(N not in N_done):
                    N_done.append(N)
                else:
                    NN_cpt+=1
                    continue
                for f in f_list:
                    for loss in loss_list:
                        for grad_optimizer in grad_optimizer_list:
                            for learning_rate in learning_rate_list:
                                for n_iter in n_iter_list:
                                    #On sélectionne le gpu aléatoirement s'il y en a plusieurs
                                    if(isinstance(num_gpus,list)):
                                        num_gpu = np.random.choice(num_gpus)
                                    else:
                                        num_gpu = num_gpus
                                    #On ne peut pas utiliser le loss hinge avec la fonction sigmoid car elle
                                    #est centrée en 0 et 1
                                    if(f==tf.sigmoid and loss=="hinge"):
                                        f=tf.tanh
                                    #Nous devons utiliser la sigmoid dans le cas d'un coup cross entropy
                                    if(f!=tf.sigmoid and loss=='cross_entropy'):
                                        f=tf.sigmoid
                                    #Nous modifions les paramètres de type tensorflow en string pour que spark ne fasse
                                    #pas d'erreur de comprehension lors de la parallelisation
                                    f_bis, grad_optimizer_bis = self.tf_to_from_str(f, grad_optimizer)
                                    #Ajout des paramètres à la liste résultat
                                    list_params.append([True, C, map(int, N), f_bis, loss, grad_optimizer_bis, learning_rate, n_iter,
                                                    batch_train, batch_valid, early_stopping,
                                                    p_early_stopping, n_threads, num_gpu]) #display=False dans le grid search
                                    #Test d'arrête si nous avons construit suffisamment de réseaux
                                    NN_cpt+=1
                                    if(NN_cpt >= n_NN):
                                        return list_params
        return list_params
    
    """
    run_NN_spark : Apprend les NN en distribué sur spark
    """
    def run_NN_spark(self, sqlContext, n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, 
                                n_iter_list, batch_train, batch_valid, early_stopping, p_early_stopping,
                                n_threads, num_gpus):
        #Création de la liste des paramètres
        list_params=self.create_list_params(n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, 
                                n_iter_list, batch_train, batch_valid, early_stopping, p_early_stopping,
                                n_threads, num_gpus)
        rdd_params = sqlContext.createDataFrame(list_params)
        res_run = rdd_params.map(lambda x: self.fit_spark(*x)).collect()
        #On met à jour la best_accuracy, la best_struct et le best_params de la classe si ils ont été battus
        for res in res_run:
            self.results.append(res)
            if(self.best_accuracy < res[0]):
                self.best_accuracy = res[0]
                self.best_struct = res[1]
                self.best_params = res[2]
                #RAJOUTER LA SAUVEGARDE DU DICTIONNAIRE
                # Si on doit enregistrer le modèle alors on le fait à l'endroit spécifié
                if(self.path_save_best_model is not None):
                    f_saver = shelve.open(self.path_save_best_model)
                    f_saver['best_accuracy'] = self.best_accuracy
                    f_saver['best_struct'] = self.best_struct
                    f_saver.close() 
    
    """
    tf_to_from_str: Etant donné que spark ne peut pasparalléliser des variables de type tensorflow il faut passer par cette
    fonction pour convertir nos variables selon le besoin qu'on a (tf vers string ou string vers tf)
    """
    def tf_to_from_str(self, f, grad_optimizer):
        dico = {tf.train.GradientDescentOptimizer:"GradientDescentOptimizer",
                tf.train.AdagradOptimizer:"AdagradOptimizer",
                tf.train.AdamOptimizer:"AdamOptimizer",
                tf.train.MomentumOptimizer:"MomentumOptimizer",
                tf.train.FtrlOptimizer:"FtrlOptimizer",
                tf.train.GradientDescentOptimizer:"GradientDescentOptimizer",
                tf.nn.relu:"relu",
                tf.sigmoid:"sigmoid",
                tf.tanh:"tanh",
                "GradientDescentOptimizer":tf.train.GradientDescentOptimizer,
                "AdagradOptimizer":tf.train.AdagradOptimizer,
                "AdamOptimizer":tf.train.AdamOptimizer,
                "MomentumOptimizer":tf.train.MomentumOptimizer,
                "FtrlOptimizer":tf.train.FtrlOptimizer,
                "GradientDescentOptimizer":tf.train.GradientDescentOptimizer,
                "relu":tf.nn.relu,
                "sigmoid":tf.sigmoid,
                "tanh":tf.tanh}
        return dico[f], dico[grad_optimizer]

    """
    fit_spark : idem à fit mais légèrement modifié pour que spark fonctionne correctement.
    """
    def fit_spark(self, verif=False, C=1, N=[1], f="sigmoid", loss='mse', grad_optimizer="GradientDescentOptimizer", learning_rate=1e-3, n_iter=1000,
            batch_train=1, batch_valid=np.Inf, early_stopping=False, p_early_stopping=3,
            n_threads=1, num_gpu=0):
        #Nous changeons f et grad_optimizer qui ont été passé en string pour que spark ne renvoie pas d'erreur
        f, grad_optimizer = self.tf_to_from_str(f, grad_optimizer)
        
        #Nous déterminons quelles sorties nous allons utiliser (cas tanh, sigmoid ou cross entropy)
        y = self.set_output(f, loss)
        output_dim = self.set_output_dim(loss)
        
        #Configuration de la session du graphe de tensorflow
        config = tf.ConfigProto()
        #config.gpu_options.allocator_type = 'BFC'
        config.allow_soft_placement=True
        config.intra_op_parallelism_threads=n_threads
        config.inter_op_parallelism_threads=n_threads
        #Définition de la session
        with tf.Session(config=config) as sess:
            #Définition du gpu à utiliser
            with tf.device('/gpu:'+str(num_gpu)):
                #Défintion de l'entrée du réseau 
                X = tf.placeholder(tf.float32, shape=[None, self.n_dim], name="X_input")
                y_ = tf.placeholder(tf.float32, shape=[None, output_dim], name="y_input")
                #Création du dictionnaire représentant le réseau de neurones
                NN = self.create_NN(X, C, N, f, loss, output_dim)

                #Formule du coût à optimiser
                if(loss=='mse'):
                    l=tf.reduce_mean(tf.square(NN["y"+str(C-1)+str(N[C-1]-1)]-y_))
                elif(loss=='hinge'):
                    l=tf.reduce_mean(tf.maximum(0.0, 1-NN["y"+str(C-1)+str(N[C-1]-1)]*y_))
                else: #cas cross entropy
                    l=-tf.reduce_sum(y_*tf.log(NN["y"+str(C-1)+str(N[C-1]-1)]))
    
                #Optimisation par descente de gradient
                train_step = self.define_grad_optimizer(grad_optimizer, learning_rate).minimize(l)
                
                #Calcul de la prédiction et de l'accuracy
                if(f==tf.tanh and (loss=='mse' or loss=='hinge')):
                    correct_prediction = tf.equal(tf.sign(NN["y"+str(C-1)+str(N[C-1]-1)]), y_)
                elif(f==tf.sigmoid and (loss=='mse' or loss=='hinge')):
                    correct_prediction = tf.equal(tf.cast(NN["y"+str(C-1)+str(N[C-1]-1)]>0.5, tf.float32), y_) #Recentrage entre 0 et 1 pour le cas sigmoid
                else:
                    correct_prediction = tf.equal(tf.argmax(NN["y"+str(C-1)+str(N[C-1]-1)],1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
                #Calcul du nombre d'évaluation à effectuer lors de l'apprentissage
                if(self.n_eval<=1 and self.n_eval>=0):
                    n_eval = int(n_iter/(self.n_eval*100))
                else:
                    n_eval = int(np.abs(self.n_eval))
                    
                #On regarde si les batch ne pas trop grand
                if(batch_train>self.n_train):
                    batch_train = self.n_train
                if(batch_valid>self.n_valid):
                    batch_valid = self.n_valid
                
                #itération, old_loss et new_loss pour déterminer l'arrête de l'appentissage ou non
                iteration=0
                acc =-1
                
                #Utile pour le early stopping
                if(early_stopping):
                    i=0
                    j=0
                    v=np.Inf
                    teta_star = NN
                    i_star=i
                
                #Initialisation de toutes nos variable contenu dans le dictionnaire
                sess.run(tf.initialize_all_variables())
                
                #On append jusqu'à convergence ou un nombre maximum d'itération
                should_stop = False
                while(iteration<=n_iter and not should_stop):
                    #Calcul de l'accuracy
                    if(iteration % n_eval == 0 and verif):
                        #On choisit aléatoirement un certain nombre de données de validation
                        xi = np.random.choice(self.ind_valid, batch_valid, replace=False)
                        #Nous remplissons les placeholders
                        feed = {X: self.X[xi].reshape((batch_valid,self.X.shape[1])), y_: y[xi].reshape((batch_valid, output_dim))}
                        #On calcul a la fois l'accuracy et le loss
                        #l'accuracy permet de voir si le modèle est performant
                        #le loss est un contrôle pour voir si la descente de gradient fonctionne correctement
                        result = sess.run([accuracy, l], feed_dict=feed)
                        acc = result[0]
                        loss_res = result[1]
                        #Nous déterminons si nous devons stopper l'apprentissage
                        if(early_stopping):
                            should_stop, teta, i, j, v, teta_star, i_star = self.early_stopping(p_early_stopping, acc, n_eval, NN, i, j, v, teta_star, i_star)
                    #Aprentissage du modèle
                    else:
                        #On choisit aléatoirement un certain nombre de données de validation (en général 1 pour la méthode stochastic)
                        xi = np.random.choice(self.ind_train, batch_train, replace=False)
                        #Nous remplissons les placeholders
                        feed = {X: self.X[xi].reshape((batch_train,self.X.shape[1])), y_: y[xi].reshape((batch_train, output_dim))}
                        #Enfin on apprend le réseau et on met les poids à jour
                        sess.run(train_step, feed_dict=feed)
                    iteration+=1
                #On enregistre le résultat du NN
                self.results.append([acc, {'C':C, 'N':N, 'f':f, 'loss':loss, 'learning_rate':learning_rate, 'grad_optimizer': grad_optimizer, 'n_iter':n_iter}])
                params={}
                for key, value in NN.iteritems():
                    if(key[0]=='y'):continue
                    params[key] = sess.run(value)
        return [acc, {'C':C, 'N':N, 'f':f, 'loss':loss, 'learning_rate':learning_rate, 'grad_optimizer': grad_optimizer, 'n_iter':n_iter}, params]
                        
    """
    fit_structure : Calcule la structure otpimale de réseau en effectuant un grid search selon
    les différents paramètres passés en arguments.
    
    Entrées:
        X : array, Données

        y : array, Sortie

        parameters : dictionnaire des paramètres pouvant contenir :
            'C': liste de C, default=[1]
            'N': liste de N, default=[1]
            'f': liste de f, default=sigmoid
            'loss': liste de loss, default='mse'
            'learning_rate': liste d'learning_rate, default=1e-3
            'n_iter': liste de n_iter, default=1000
            'grad_optimizer': liste de methodes de descente de gradient, default=tf.train.GradientDescentOptimizer
            
        n_NN : int, default = 1, Nombre de réseau que l'on veut créer aléatoirement et entrainer.

        p_train : float, default=0.8, Pourcentage de données à utiliser en train.
        
        p_valid : float, default=None Pourcentage de données à utiliser en validation. Si none alors
            => n_valid = 1-n_train

        batch_train : int, default=1 (méthode stochastique). Nombre de données passées à l'étape de train.

        batch_valid : float ou int, default=None (toutes les données de validation sont utilisés à chaque
            calcul de l'accuracy. Nombre de données sur les données de validation passées au moment du calcul de 
            l'accuracy (en pourcentage ou nombre de données).

        shuffle : boolean, default=False. Si vrai, mélange les données avant l'apprentissage.
            Si une DataFrame est passée en paramètre le traitement est rapide, si c'est un np.array c'est
            plus lent.
            
        early_stopping : boolean, deafault=False. Utilisation ou non  de l'algo early stopping pour arrêter
            prématurément le training.

        p_early_stopping : int, deafault=3. Patience de l'algorithme early stopping. Si early_stopping=False,
            alors ce paramètre est inutile.
        
        multithreading : boolean, default=False. Si vrai, va calculer les différents réseaux en 
            parallèles, sinon ne multithread pas.

        n_threads : int, default=1. Nombre de thread que l'on veut utiliser pour
            nos opérations. Si pas supérieur à 0 alors on ne change rien.

        num_gpus : int ou list de int, default=0, le numéro du gpu à utiliser. Le gpu 0 est utilisé par défaut,
            cependant si la machine n'a pas de gpu, l'option allow_soft_placement permet d'éviter
            la collision de device.
            Si une liste est donnée, alors chacun des réseau va etre calculé sur alternativement sur 
            les différents gpu dans la liste
        
        one_hot : boolean, default=True, Si vrai alors on crée des sorties one hot, si faux, alors
            c'est que les sorties passées en paramètre sont déjà one hot (et que l'apprentissage avec
            des loss mse et hinge n'est pas possible)
        
        p_show : float, default=0.2, pourcentage d'affichage par rapport au nombre de réseaux créés. Permet de voir 
            l'avancement de l'apprentissage de structure. (uniquement pour le cas non multithread)
        
        sc : sparkcontext, default=None, si définit alors va lancer le grid search sur spark. Il faut de plus
            définir le sqlContext, les deux vont ensembles.
        
        sqlContext : sqlContext, default=None, si définit alors va lancer le grid search sur spark.
        
    Sorties :
        Aucune 
    """
    
    def fit_structure(self, X, y, parameters, n_NN=1, p_train=0.8, p_valid=None, 
                      batch_train=1, batch_valid=None, shuffle=True, early_stopping=False,
                      p_early_stopping=3, multithreading=False,
                      n_threads=1, num_gpus=0, one_hot=True, p_show=0.2, sc=None, sqlContext=None):
        
        #Mélange des données
        """
        if(shuffle==True):
            if(isinstance(X, pd.DataFrame) and (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame))):
                new_index = np.random.permutation(X.index)
                X = X.reindex(new_index)
                y = y.reindex(new_index)
            else:
                X, y = utils.shuffle(np.array(X), np.array(y))
        """
        #Nous remettons sous forme de tableau les données si elles ne le sont pas déjà,
        #Nous déterminons également la taille et dimensions des données de train et de validation
        self.fit_X_y(X, y, p_train, p_valid, one_hot)
        
        #Nous recastons en int les paramètres devant être des int pour éviter les erreurs
        #ainsi que les warnings
        n_NN = int(n_NN)
        batch_train = int(batch_train)
        n_theads = int(n_threads)
        if(isinstance(num_gpus, list)):
            num_gpus = map(int, num_gpus)
        else:
            num_gpus = int(num_gpus)
        if(n_threads<=0):
            n_threads=1
            
        #Si pas de batch_val défini alors on sélectionne toutes les données
        if(batch_valid is None):
            batch_valid=self.n_valid
        elif(isinstance(batch_valid, int)):
            if(batch_valid>self.n_valid):
                batch_valid=self.n_valid
            else:
                batch_valid = batch_valid
        else:
            batch_valid = int(self.n_valid*batch_valid)
        
        #Récupération des paramètres
        if('C' in parameters and 'N' in parameters):
            C_list = parameters['C']
            N_list = parameters['N']
        else:
            C_list = [1]
            N_list = [1]
        if('f' in parameters):
            f_list = parameters['f']
        else:
            f_list = [tf.sigmoid]
        if('learning_rate' in parameters):
            learning_rate_list = parameters['learning_rate']
        else:
            learning_rate_list = [1e-3]
        if('loss' in parameters):
            loss_list = parameters['loss']
        else:
            loss_list = 'mse'
        if('n_iter' in parameters):
            n_iter_list = parameters['n_iter']
        else:
            n_iter_list = [1000]
        if('grad_optimizer' in parameters):
            grad_optimizer_list = parameters['grad_optimizer']
        else:
            grad_optimizer_list = [tf.train.GradientDescentOptimizer]

        self.trained=True
        
        #Cas spark
        if(sc is not None):
            if(sqlContext is None):
                print "Il faut aussi passer en paramètre le sqlContext"
                return
            sc.broadcast(self.X)
            sc.broadcast(self.y)
            self.run_NN_spark(sqlContext, n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, 
                            n_iter_list, batch_train, batch_valid, early_stopping, p_early_stopping,
                            n_threads, num_gpus)
                   
            return
        
        #Cas multithreadé
        if(multithreading==True):
            self.run_NN_multi_local(n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, 
                                        n_iter_list, batch_train, batch_valid, early_stopping, p_early_stopping,
                                        n_threads, num_gpus) 
        #Cas non multithreadé
        else:
            self.run_NN(n_NN, C_list, N_list, f_list, loss_list, grad_optimizer_list, learning_rate_list, 
                        n_iter_list, batch_train, batch_valid, early_stopping, p_early_stopping,
                        n_threads, num_gpus, p_show)
        
    
