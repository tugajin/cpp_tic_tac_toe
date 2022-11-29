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
    m.def("from_hash",&game::from_hash);
    
    py::class_<movelist::MoveList>(m, "MoveList")
        .def(py::init<>())
        .def("init", &movelist::MoveList::init)
        .def("add", &movelist::MoveList::add)
        .def("begin", &movelist::MoveList::begin)
        .def("end", &movelist::MoveList::end)
        .def("len", &movelist::MoveList::len)
        .def("__str__", &movelist::MoveList::str)
        .def("__getitem__", &movelist::MoveList::operator[]);

    py::class_<game::Position>(m, "Position")
        .def(py::init<>())
        .def("turn", &game::Position::turn)
        .def("self", &game::Position::self)
        .def("enemy", &game::Position::enemy)
        .def("next",&game::Position::next)
        .def("__str__",&game::Position::str)
        .def("has_win",&game::Position::has_win)
        .def("is_draw",&game::Position::is_draw)
        .def("is_lose",&game::Position::is_lose)
        .def("is_done",&game::Position::is_done)
        .def("hash_key",&game::Position::hash_key)
        .def("legal_moves",&game::Position::legal_moves)
        .def("feature",&game::Position::feature);

    return m.ptr();
}