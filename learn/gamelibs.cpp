#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include "../ai/game.hpp"

namespace py = pybind11;
TeeStream Tee;

PYBIND11_PLUGIN(gamelibs) {
    
    py::module m("gamelibs", "gamelibs made by pybind11");

    py::enum_<Color>(m, "Color", py::arithmetic())
        .value("BLACK", Color::BLACK)
        .value("WHITE", Color::WHITE);

    py::enum_<Move>(m, "Move", py::arithmetic())
        .value("MOVE_NONE", Move::MOVE_NONE);

    m.def("val_to_move",[](int m){
        return Move(m);
    });
    m.def("move_to_val",[](Move x){
        return static_cast<int>(x);
    });

    py::class_<MoveList>(m, "MoveList")
        .def(py::init<>())
        .def("init", &MoveList::init)
        .def("add", &MoveList::add)
        .def("begin", &MoveList::begin)
        .def("end", &MoveList::end)
        .def("len", &MoveList::len)
        .def("__str__", &MoveList::str)
        .def("__getitem__", &MoveList::operator[]);

    py::class_<Position>(m, "Position")
        .def(py::init<>())
        .def("turn", &Position::turn)
        .def("self", &Position::self)
        .def("enemy", &Position::enemy)
        .def("next",&Position::next)
        .def("__str__",&Position::str)
        .def("has_win",&Position::has_win)
        .def("is_draw",&Position::is_draw)
        .def("is_lose",&Position::is_lose)
        .def("is_done",&Position::is_done)
        .def("hash_key",&Position::hash_key)
        .def("legal_moves",&Position::legal_moves)
        .def("feature",&Position::feature);

    return m.ptr();
}