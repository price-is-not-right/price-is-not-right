(define (domain hanoi)
  (:requirements :strips :typing)
  (:types disk peg - location)

  (:predicates
    (on ?disk - disk ?location - location)  ; disk is on another disk or a peg
    (clear ?location - location)       ; no disk is on disk
    (grasped ?disk - disk)     ; robot is grasped a disk
    (smaller ?disk - disk ?location - location) ; disk is smaller than the location
    (free-gripper) ; the gripper is free
  )

  (:action pick
    :parameters (?d - disk ?l - location)
    :precondition (and (clear ?d) (on ?d ?l) (free-gripper))
    :effect (and (clear ?l) (not(on ?d ?l)) (grasped ?d) (not (free-gripper))))

  (:action place
    :parameters (?d - disk ?l - location)
    :precondition (and (grasped ?d) (smaller ?d ?l) (clear ?l))
    :effect (and (not (grasped ?d)) (on ?d ?l) (not(clear ?l)) (free-gripper)))
)