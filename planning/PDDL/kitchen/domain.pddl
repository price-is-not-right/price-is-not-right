(define (domain kitchen)
  (:requirements :strips :typing)
  (:types object - entity
          stove button pot bread storage table - object)
  
  (:predicates
    (on ?obj1 - object ?obj2 - object) ; object is at a object
    (grasped ?obj - object)             ; agent is grasped an object
    (stove_on)                          ; stove is turned on
    (free-gripper)                      ; the gripper is free
    (cooked ?obj - object)             ; the object is cooked
  )
  
  (:action pick
    :parameters (?o - object ?l - object)
    :precondition (and (not (grasped ?o)) (on ?o ?l) (free-gripper))
    :effect (and (grasped ?o) (not (on ?o ?l)) (not (free-gripper))))
  
  (:action place
    :parameters (?o - object ?l - object)
    :precondition (grasped ?o)
    :effect (and (on ?o ?l) (not (grasped ?o)) (free-gripper)))

  (:action turn-on-stove
    :parameters ()
    :precondition (and (not (stove_on)) (free-gripper))
    :effect (and (stove_on) ))
  
  (:action turn-off-stove
    :parameters ()
    :precondition (and (stove_on) (free-gripper))
    :effect (and (not (stove_on))))

  (:action cook
    :parameters (?o - object)
    :precondition (and (stove_on) (on ?o pot) (on pot stove))
    :effect (cooked ?o)
  )
)
