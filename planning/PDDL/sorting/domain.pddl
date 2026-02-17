(define (domain sorting)
  (:requirements :strips :typing)
  (:types object box)

  (:predicates
    (on ?obj - object ?box - box)  ; object is in the box
    (clear ?obj - object)          ; object is not being held
    (grasped ?obj - object)        ; agent is grasped an object
    (matches ?obj - object ?box - box) ; object matches the box
    (free-gripper) ; the gripper is free
  )

  (:action pick
    :parameters (?o - object)
    :precondition (and (clear ?o) (free-gripper))
    :effect (and (not (clear ?o)) (grasped ?o) (not (free-gripper))))

  (:action place
    :parameters (?o - object ?b - box)
    :precondition (and (grasped ?o) (clear ?b) (matches ?o ?b))
    :effect (and (not (grasped ?o)) (on ?o ?b) (not (clear ?b)) (free-gripper)))
)
