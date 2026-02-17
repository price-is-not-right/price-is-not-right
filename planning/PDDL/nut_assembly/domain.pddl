(define (domain nut_assembly)
  (:requirements :strips :typing)
  (:types nut peg table - object
          round_nut square_nut - nut
          round_peg square_peg - peg)

  (:predicates
    (on ?nut - nut ?o - object)  ; nut is on the peg
    (clear ?o - object)          ; peg is empty
    (grasped ?nut - nut)        ; agent is grasped a nut
    (matches ?nut - nut ?peg - peg) ; nut matches the peg type
    (free-gripper) ; the gripper is free
  )

  (:action pick
    :parameters (?n - nut ?o - object)
    :precondition (and (clear ?n) (free-gripper) (on ?n ?o))
    :effect (and (not (clear ?n)) (grasped ?n) (not (on ?n ?o)) (not (free-gripper)))
  )

  (:action place
    :parameters (?n - nut ?p - peg)
    :precondition (and (grasped ?n) (clear ?p) (matches ?n ?p))
    :effect (and (not (grasped ?n)) (on ?n ?p) (not (clear ?p)) (free-gripper))
    )
)
