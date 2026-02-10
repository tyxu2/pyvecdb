#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "IndexFlat.h"
#include "IndexIVF.h"
#include "IndexHNSW.h"
#include "Distance.h"

namespace py = pybind11;
using namespace pyvecdb;

// Helper to wrap add
void index_add(Index& index, py::array_t<float> x) {
    py::buffer_info buf = x.request();
    if (buf.ndim != 2) throw std::runtime_error("Number of dimensions must be two");
    if (buf.shape[1] != index.get_d()) throw std::runtime_error("Dimension mismatch");

    index.add(buf.shape[0], static_cast<float*>(buf.ptr));
}

// Helper to wrap train
void index_train(Index& index, py::array_t<float> x) {
    py::buffer_info buf = x.request();
    if (buf.ndim != 2) throw std::runtime_error("Number of dimensions must be two");
    if (buf.shape[1] != index.get_d()) throw std::runtime_error("Dimension mismatch");

    index.train(buf.shape[0], static_cast<float*>(buf.ptr));
}

// Helper to wrap search
std::pair<py::array_t<float>, py::array_t<long>> index_search(Index& index, py::array_t<float> x, int k) {
    py::buffer_info buf = x.request();
    if (buf.ndim != 2) throw std::runtime_error("Number of dimensions must be two");
    if (buf.shape[1] != index.get_d()) throw std::runtime_error("Dimension mismatch");

    int n = buf.shape[0];

    auto distances = py::array_t<float>({n, k});
    auto labels = py::array_t<long>({n, k});

    index.search(n, static_cast<float*>(buf.ptr), k,
                 static_cast<float*>(distances.request().ptr),
                 static_cast<long*>(labels.request().ptr));

    return {distances, labels};
}

PYBIND11_MODULE(_pyvecdb, m) {
    m.doc() = "Lightweight vector database C++ core";
    m.def("is_cuda_enabled", &is_cuda_enabled, "Check if compiled with CUDA support");

    py::class_<Index, std::shared_ptr<Index>>(m, "Index")
        .def("get_d", &Index::get_d)
        .def("get_ntotal", &Index::get_ntotal)
        .def("add", &index_add)
        .def("search", &index_search)
        .def("train", &index_train);

    py::class_<IndexFlat, Index, std::shared_ptr<IndexFlat>>(m, "IndexFlat")
        .def(py::init<int>())
        .def("reset", &IndexFlat::reset);

    py::class_<IndexIVF, Index, std::shared_ptr<IndexIVF>>(m, "IndexIVF")
        .def(py::init<int, int>())
        .def("set_nprobe", &IndexIVF::set_nprobe);

    py::class_<IndexHNSW, Index, std::shared_ptr<IndexHNSW>>(m, "IndexHNSW")
        .def(py::init<int, int, int>(), py::arg("d"), py::arg("M")=16, py::arg("efConstruction")=200)
        .def("set_ef", &IndexHNSW::set_ef);
}
