add_library(svm_common svm_common.c)
add_library(svm_learn_main svm_learn_main.c)
add_library(svm_learn svm_learn.c)
add_library(svm_classify svm_classify.c)
add_library(svm_hideo svm_hideo.c)
target_link_libraries(svm_classify svm_common)
target_link_libraries(svm_hideo svm_common)
target_link_libraries(svm_learn svm_hideo -lm)