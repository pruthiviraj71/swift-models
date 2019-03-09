import TensorFlow
import Python

# %include "EnableIPythonDisplay.swift"
IPythonDisplay.shell.enable_matplotlib("inline")

let np = Python.import("numpy")
let h5py = Python.import("h5py")
let plt = Python.import("matplotlib.pyplot")


func load_dataset()->(PythonObject, PythonObject, PythonObject, PythonObject, PythonObject){
    let train_dataset = h5py.File("train_catvnoncat.h5","r")
    let train_set_x_orig = np.array(train_dataset["train_set_x"])
    let train_set_y_orig = np.array(train_dataset["train_set_y"])

    let test_dataset = h5py.File("test_catvnoncat.h5", "r")
    let test_set_x_orig = np.array(test_dataset["test_set_x"])
    let test_set_y_orig = np.array(test_dataset["test_set_y"])
    let classes = np.array(test_dataset["list_classes"])
    
    let train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    let test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    var train_set_x = train_set_x_flatten / 255
    var test_set_x = test_set_x_flatten / 255
    
    return (train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, classes)
}

//Sigmoid Function
func sigmoid(z:PythonObject)->(PythonObject){
    return (1/(1+np.exp(-z)))
}

//Forward and Backward Propagation
func propagate(){
    
    let (train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, classes) = load_dataset()
    
    let X = train_set_x
    let b = 0
    var w = np.zeros(12288)
    w = w.reshape(12288,1)
    let A = sigmoid(z:(np.dot(w.T, X)+b))
    
}