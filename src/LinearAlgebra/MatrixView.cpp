#include "../include/LinearAlgebra/MatrixView.h"

using namespace SharedMath::LinearAlgebra;

MatrixView::MatrixView(AbstractMatrix* matrix_) : matrix(matrix_)
{
    if(!matrix){
        throw std::invalid_argument("Matrix is nullptr in MatrixView");
    }
    StartRowIdx = 0;
    StartColIdx = 0;
    EndRowIdx = matrix->rows();
    EndColIdx = matrix->cols();
}

MatrixView::MatrixView(size_t startRow, size_t endRow, size_t startCol, size_t endCol, AbstractMatrix* matrix_) :
    StartRowIdx(startRow), StartColIdx(startCol),
    EndRowIdx(endRow), EndColIdx(endCol),
    matrix(matrix_)    
{
    if(!matrix){
        throw std::invalid_argument("Matrix is nullptr in MatrixView");
    } 
    if(StartRowIdx < 0 || StartColIdx < 0 || EndRowIdx < 0 || EndColIdx < 0){
        throw std::invalid_argument("Some of index less than 0");
    }    
    if(EndRowIdx > matrix->rows() || EndColIdx > matrix->cols()){
        throw std::invalid_argument("EndIndeces more than matrix have");
    }
    if(StartColIdx > EndColIdx || StartRowIdx > EndRowIdx){
        throw std::invalid_argument("Start index more than end index");
    }
}


size_t MatrixView::rows() const{
    return EndRowIdx - StartRowIdx;
}

size_t MatrixView::cols() const{
    return EndColIdx - StartColIdx;
}

double MatrixView::get(size_t row, size_t col) const{
    checkIndices(row, col);
    return matrix->get(StartRowIdx + row, StartColIdx + col);
}

void MatrixView::set(size_t row, size_t col, double val){
    checkIndices(row, col);
    matrix->set(StartRowIdx + row, StartColIdx + col, val);
} 

double MatrixView::operator()(size_t row, size_t col) const{
    return get(row, col);
}

double& MatrixView::operator()(size_t row, size_t col){
    checkIndices(row, col);
    return matrix->get(StartRowIdx + row, StartColIdx + col);
}

MatrixView MatrixView::subView(size_t startRow, size_t endRow, size_t startCol, size_t endCol) const{
    if (startRow >= endRow || startCol >= endCol) {
        throw std::invalid_argument("Invalid subview dimensions");
    }
    if (endRow > rows() || endCol > cols()) {
        throw std::out_of_range("Subview dimensions exceed view bounds");
    }
    return MatrixView(
        StartRowIdx + startRow,
        StartRowIdx + endRow,
        StartColIdx + startCol,
        StartColIdx + endCol,
        matrix
    );
}

void MatrixView::checkIndices(size_t row, size_t col) const{
    if(row >= rows() || col >= cols()){
        throw std::out_of_range("MatrixView indices out of range");
    }
    if(!matrix){
        throw std::runtime_error("MatrixView is not initialized");
    }
}

